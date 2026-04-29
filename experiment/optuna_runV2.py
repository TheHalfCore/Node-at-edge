from math import exp
import sys
import os
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import optuna
import torch
import lib
from unimibV2 import UniMiBExperiment
from collections import Counter

filename = "optuna_Test_NODE_withDataAgg_results_V2.csv"
# =========================
# ✅ OPTUNA OBJECTIVE
# =========================
def objective(trial):
    try:
        exp = UniMiBExperiment(gpu_id=0, is_cpu=True, delete_logs=False)

        exp.load_and_preprocess_data()
        # hyperparameters
        exp.layer_dim = trial.suggest_int("layer_dim", 1, 48)
        exp.num_layers = trial.suggest_int("num_layers", 1, 6)
        exp.depth = trial.suggest_int("depth", 1, 6)
        batch_size = trial.suggest_categorical("batch_size", [32, 56, 64, 96])
        epochs = trial.suggest_int("epochs", 50, 60)
        lr = trial.suggest_float("lr", 1e-5, 1e-4, log=True)

        exp.epochs = epochs
        
        exp.optimizer_params = {
            'nus': (0.7, 1.0),
            'betas': (0.95, 0.998),
            'lr': lr
        }

        exp.create_model()
        print("Train samples:")
        print(Counter(exp.data.y_train))

        print("Valid samples:")
        print(Counter(exp.data.y_valid))

        print("Test samples:")
        print(Counter(exp.data.y_test))

        print(f"layer_dim: {exp.layer_dim}, num_layers: {exp.num_layers}, depth: {exp.depth}, batch_size: {batch_size}, epochs: {epochs}, lr: {lr}")
        print(f"total_params: {exp.total_params}, total_params_size_with_buffer_KB: {exp.total_params_size_with_buffer_KB}, cpu_activation_memory_KB: {exp.cpu_activation_memory_KB}, cpu_estimate_inference_size_KB: {exp.cpu_estimate_inference_size_KB}")
        exp.create_trainer()
        
        steps = 5000

        data_iter = lib.iterate_minibatches(
            exp.data.X_train,
            exp.data.y_train,
            batch_size=batch_size,
            shuffle=True,
            epochs=epochs
        )

        for step, batch in enumerate(data_iter):
            xb, yb = batch
            xb = torch.as_tensor(xb, device=exp.device)
            yb = torch.as_tensor(yb, device=exp.device)

            exp.trainer.train_on_batch(xb, yb, device=exp.device)

            if step >= steps:
                break

        f1 = exp.evaluate_f1(exp.data.X_valid, exp.data.y_valid)
        test_f1 = exp.evaluate_f1(exp.data.X_test, exp.data.y_test)
        
        log_trial({ # log the successful trial with its F1 and memory stats
            "layer_dim": exp.layer_dim,
            "num_layers": exp.num_layers,
            "depth": exp.depth,
            "lr": lr,
            "batch_size": batch_size,
            "epochs": epochs,
            "cpu_activation_memory_KB": exp.cpu_activation_memory_KB,
            "cpu_estimate_inference_size_KB": exp.cpu_estimate_inference_size_KB,
            "total_params": exp.total_params,
            "total_params_size_with_buffer_KB": exp.total_params_size_with_buffer_KB,
            "f1_score": f1,
            "test_f1_score": test_f1
        })
        
        return f1

    except RuntimeError as e: # catch OOM errors
        if "out of memory" in str(e).lower():
            print("⚠️ OOM detected, skipping trial")

            return float("-inf")  # bad score so Optuna skips it
        else:
            raise e

# =========================
# 📝 LOGGING FUNCTION
# =========================
def log_trial(result_dict):
    df = pd.DataFrame([result_dict])
    if os.path.exists(filename):
        df.to_csv(filename, mode='a', header=False, index=False)
    else:
        df.to_csv(filename, mode='w', header=True, index=False)

# =========================
# 🚀 RUN OPTUNA
# =========================
if __name__ == "__main__":
    if os.path.exists(filename):
        os.remove(filename)
    study = optuna.create_study(direction="maximize")
    n_trials = int(input("Enter number of trials [20]: "))
    study.optimize(objective, n_trials=n_trials)

    print("\n🔥 Best Results:")
    print("Best F1:", study.best_value)
    print("Best Params:", study.best_params)