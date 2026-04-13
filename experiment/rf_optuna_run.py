import optuna
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

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
        n_estimators=trial.suggest_int("n_estimators", 1, 2000),
        max_depth=trial.suggest_int("max_depth", 2, 200),
        max_features=trial.suggest_categorical(
            "max_features", ["sqrt", "log2", None]
        ),
        min_samples_split=trial.suggest_int("min_samples_split", 2, 10),
        min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 5),
        bootstrap=True,
        random_state=42,
        n_jobs=-1
    )
    ## min_samples_split = 2 default value
    ## min_samples_leaf  = 1
    rf.fit(X_train, y_train)

    y_val_pred = rf.predict(X_val)

    f1 = f1_score(y_val, y_val_pred, average="macro")

    return f1


# =========================
# 🚀 RUN OPTUNA
# =========================
if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    n_trials = int(input("Enter number of trials [30]: "))
    study.optimize(objective, n_trials=n_trials)

    print("\n🔥 Best params:", study.best_params)
    print("Best F1:", study.best_value)
    print("Best Trial Number:", study.best_trial.number)
    print("Best Trial Params:", study.best_trial.params)


    # # =========================
    # # 🏆 TRAIN FINAL MODEL
    # # =========================
    # best = study.best_params

    # rf = RandomForestClassifier(
    #     **best,
    #     bootstrap=True,
    #     random_state=42,
    #     n_jobs=-1
    # )

    # rf.fit(X_train, y_train)

    # y_test_pred = rf.predict(X_test)

    # print("\n✅ FINAL TEST F1 (macro):", f1_score(y_test, y_test_pred, average="macro"))