import optuna
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
import pickle
import sys
import psutil
import os
from memory_profiler import memory_usage

filename = "optuna_Test_RF_withDataAgg_results.csv"
# =========================
# 📂 LOAD DATA (same as yours)
# =========================
train = pd.read_csv("../notebooks/data/UniMiB/unimib_train.csv")
val   = pd.read_csv("../notebooks/data/UniMiB/unimib_val.csv")
test  = pd.read_csv("../notebooks/data/UniMiB/unimib_test.csv")

def aggregate_windows(df):
    agg = df.groupby("ID").agg({
        "ax": ["mean", "std", "min", "max"],
        "ay": ["mean", "std", "min", "max"],
        "az": ["mean", "std", "min", "max"],
        "mag": ["mean", "std", "min", "max"],
        "label": "first"
    })

    agg.columns = [
        "_".join(col) if isinstance(col, tuple) else col
        for col in agg.columns
    ]

    return agg.reset_index(drop=True)

def get_model_size_mb(model):  # Returns the size of the model in MB
    size_bytes = len(pickle.dumps(model))
    return size_bytes / (1024 ** 2)

process = psutil.Process(os.getpid())

def get_memory_mb(): # Returns the current memory usage of the process in MB
    return process.memory_info().rss / (1024 ** 2)

def log_trial(result_dict, filename=filename):
    df = pd.DataFrame([result_dict])

    if os.path.exists(filename):
        df.to_csv(filename, mode='a', header=False, index=False)
    else:
        df.to_csv(filename, mode='w', header=True, index=False)

train_agg = aggregate_windows(train)
val_agg   = aggregate_windows(val)
test_agg  = aggregate_windows(test)

X_train = train_agg.drop(columns=["label_first"])
y_train = train_agg["label_first"]

X_val = val_agg.drop(columns=["label_first"])
y_val = val_agg["label_first"]

X_test = test_agg.drop(columns=["label_first"])
y_test = test_agg["label_first"]


# =========================
# 🎯 OPTUNA OBJECTIVE
# =========================
def objective(trial):
    
    rf = RandomForestClassifier(
        n_estimators=trial.suggest_int("n_estimators", 1, 300),
        max_depth=trial.suggest_int("max_depth", 2, 30),
        max_features=trial.suggest_categorical(
            "max_features", [None]
        ),
        # min_samples_split=trial.suggest_int("min_samples_split", 2, 10),
        # min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 5),
        bootstrap=True,
        random_state=42,
        n_jobs=-1
    )
    ## min_samples_split = 2 default value
    ## min_samples_leaf  = 1
    def train_model():
        rf.fit(X_train, y_train)
        return rf
    
    # Memory usage
    train_mem_usage, rf = memory_usage((train_model,), retval=True, interval=0.1, max_usage=True) # Get the peak memory usage during training
    model_mem = get_model_size_mb(rf)
    
    #Validation
    y_val_pred = rf.predict(X_val)
    #F1 score
    f1 = f1_score(y_val, y_val_pred, average="macro")

    test_f1 = f1_score(y_test, rf.predict(X_test), average="macro")
    
    # Log memory usage as user attributes in Optuna
    trial.set_user_attr("model_size_mb", model_mem)
    trial.set_user_attr("training_mem_mb", train_mem_usage)

    print(f"\nTrial {trial.number}")
    print(f"F1: {f1:.4f}")
    print(f"Model size: {model_mem:.2f} MB")
    print(f"Training memory delta: {train_mem_usage:.2f} MB")
    
    log_trial({ # log the successful trial with its F1 and memory stats
            "n_estimators": rf.n_estimators,
            "max_depth": rf.max_depth,
            "max_features": rf.max_features,
            "min_samples_split": rf.min_samples_split,
            "min_samples_leaf": rf.min_samples_leaf,
            "model_size_mb": model_mem,
            "training_mem_mb": train_mem_usage,
            "f1_score": f1,
            "test_f1_score": test_f1
        })
    
    return f1, model_mem  # We want to maximize F1 and minimize model size
        

# =========================
# 🚀 RUN OPTUNA
# =========================
if __name__ == "__main__":
    if os.path.exists(filename):
        os.remove(filename)
    study = optuna.create_study(directions=["maximize", "minimize"]) # We want to maximize F1 and minimize model size
    n_trials = int(input("Enter number of trials [30]: "))
    study.optimize(objective, n_trials=n_trials)

    best_trial = None
    best_f1 = -1

    for t in study.trials:
        f1 = t.values[0]
        if f1 > best_f1:
            best_f1 = f1
            best_trial = t

    print("\n🎯 Best under constraint:")
    print(best_trial.params)
    
    # =========================
    # 📊 ANALYZE RESULTS
    # =========================
    print("\n🔥 Pareto-optimal trials:")
    for t in study.best_trials: # These are the Pareto-optimal trials (best trade-offs between F1 and model size)
        print(f"\nTrial {t.number}")
        print("  Params:", t.params)
        print("  F1:", t.values[0])        # maximize
        print("  Model size:", t.values[1]) # minimize

    print("\n📊 All trials with metrics and user attributes:")
    for t in study.trials:
        print(f"Trial {t.number}")
        print("  Params:", t.params)
        print("  F1:", t.values[0])
        print("  Model size (MB):", t.values[1])
        print("  Training mem (MB):", t.user_attrs.get("training_mem_mb"))
    
    # # =========================
    # # 🏆 TRAIN FINAL MODEL
    # # =========================
    # best = study.best_trials[0].params  # Get the params of the best trial (you can also choose based on a trade-off)

    # rf = RandomForestClassifier(
    #     **best,
    #     bootstrap=True,
    #     random_state=42,
    #     n_jobs=-1
    # )

    # rf.fit(X_train, y_train)

    # y_test_pred = rf.predict(X_test)

    # print("\n✅ FINAL TEST F1 (macro):", f1_score(y_test, y_test_pred, average="macro"))