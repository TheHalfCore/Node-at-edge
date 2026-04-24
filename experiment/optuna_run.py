import sys
import os
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import optuna
import torch
import lib
from unimib import UniMiBExperiment  # your file

# =========================
# ✅ OPTUNA OBJECTIVE
# =========================
def objective(trial):
    try:
        exp = UniMiBExperiment(gpu_id=0)
        exp.device = "cuda"   # using GPU
        
        torch.cuda.empty_cache()  # clear GPU memory before starting

        exp.load_and_preprocess_data()

        # hyperparameters
        exp.layer_dim = trial.suggest_int("layer_dim", 1, 64)
        exp.num_layers = trial.suggest_int("num_layers", 1, 8)
        exp.depth = trial.suggest_int("depth", 1, 4)

       # lr = trial.suggest_float("lr", 1e-4, 3e-3, log=True)

        exp.optimizer_params = {
            'nus': (0.7, 1.0),
            'betas': (0.95, 0.998),
            #'lr': lr
        }

        exp.create_model()
        exp.create_trainer()

        batch_size = 1024 * 16
        steps = 5000

        data_iter = lib.iterate_minibatches(
            exp.data.X_train,
            exp.data.y_train,
            batch_size=batch_size,
            shuffle=True,
            epochs=10
        )

        for step, batch in enumerate(data_iter):
            xb, yb = batch
            xb = torch.as_tensor(xb, device=exp.device)
            yb = torch.as_tensor(yb, device=exp.device)

            exp.trainer.train_on_batch(xb, yb, device=exp.device)

            if step >= steps:
                break

        f1 = exp.evaluate_f1(exp.data.X_valid, exp.data.y_valid)

        torch.cuda.reset_peak_memory_stats()  # reset peak memory stats
        
        log_trial({ # log the successful trial with its F1 and memory stats
            "layer_dim": exp.layer_dim,
            "num_layers": exp.num_layers,
            "depth": exp.depth,
            "alloc_memory": exp.gpu_alloc,
            "res_memory": exp.gpu_reserved,
            "max_alloc_memory": exp.gpu_peak,
            "f1_score": f1
        })
        
        return f1

    except RuntimeError as e: # catch OOM errors
        if "out of memory" in str(e).lower():
            print("⚠️ OOM detected, skipping trial")

            log_trial({ # log the failed trial with -1 for memory and f1
                "layer_dim": exp.layer_dim,
                "num_layers": exp.num_layers,
                "depth": exp.depth,
                "alloc_memory": -1,
                "res_memory": -1,
                "max_alloc_memory": -1,
                "f1_score": float("-inf")
            })
            
            torch.cuda.empty_cache()  # VERY IMPORTANT

            return float("-inf")  # bad score so Optuna skips it
        else:
            raise e

# =========================
# 📝 LOGGING FUNCTION
# =========================
def log_trial(result_dict, filename="optuna_NODE_results.csv"):
    df = pd.DataFrame([result_dict])

    if os.path.exists(filename):
        df.to_csv(filename, mode='a', header=False, index=False)
    else:
        df.to_csv(filename, mode='w', header=True, index=False)

# =========================
# 🚀 RUN OPTUNA
# =========================
if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    n_trials = int(input("Enter number of trials [20]: "))
    study.optimize(objective, n_trials=n_trials)

    print("\n🔥 Best Results:")
    print("Best F1:", study.best_value)
    print("Best Params:", study.best_params)


    # # =========================
    # # 🏆 TRAIN FINAL MODEL
    # # =========================
    best = study.best_params

    exp = UniMiBExperiment(gpu_id=0)
    exp.device = "cuda"

    exp.load_and_preprocess_data()

    exp.layer_dim = best["layer_dim"]
    exp.num_layers = best["num_layers"]
    exp.depth = best["depth"]

    exp.optimizer_params = {
        'nus': (0.7, 1.0),
        'betas': (0.95, 0.998),
        'lr': best["lr"]
    }

    exp.create_model()
    exp.create_trainer()

    print("\n🚀 Training final model with best params...")
    exp.train_data()

    torch.save(exp.model.state_dict(), "best_optuna_model.pt")

    print("✅ Model saved: best_optuna_model.pt")