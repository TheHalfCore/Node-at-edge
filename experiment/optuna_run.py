import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import optuna
import torch
import lib
from unimib import UniMiBExperiment  # your file

# =========================
# ✅ OPTUNA OBJECTIVE
# =========================
def objective(trial):
    # ⚠️ Use CPU to avoid GPU OOM
    exp = UniMiBExperiment(gpu_id=0)
    exp.device = "cpu"

    # Load data
    exp.load_and_preprocess_data()

    # =========================
    # 🔧 Hyperparameters to tune
    # =========================
    exp.layer_dim = trial.suggest_int("layer_dim", 48, 1024)
    exp.num_layers = trial.suggest_int("num_layers", 4, 64)
    exp.depth = trial.suggest_int("depth", 6, 30)

    lr = trial.suggest_float("lr", 1e-4, 3e-3, log=True)

    # update optimizer params
    exp.optimizer_params = {
        'nus': (0.7, 1.0),
        'betas': (0.95, 0.998),
        'lr': lr
    }

    # =========================
    # 🧠 Create model + trainer
    # =========================
    exp.create_model()
    exp.create_trainer()

    # =========================
    # 🔁 FAST TRAINING LOOP
    # =========================
    batch_size = 1024  
    steps = 200        

    data_iter = lib.iterate_minibatches(
        exp.data.X_train,
        exp.data.y_train,
        batch_size=batch_size,
        shuffle=True,
        epochs=1
    )

    for step, batch in enumerate(data_iter):
        exp.trainer.train_on_batch(*batch, device=exp.device)

        # stop early for speed
        if step >= steps:
            break

    # =========================
    # 📊 VALIDATION SCORE
    # =========================
    f1 = exp.evaluate_f1(exp.data.X_valid, exp.data.y_valid)

    return f1


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
    # best = study.best_params

    # exp = UniMiBExperiment(gpu_id=0)
    # exp.device = "cpu"

    # exp.load_and_preprocess_data()

    # exp.layer_dim = best["layer_dim"]
    # exp.num_layers = best["num_layers"]
    # exp.depth = best["depth"]

    # exp.optimizer_params = {
    #     'nus': (0.7, 1.0),
    #     'betas': (0.95, 0.998),
    #     'lr': best["lr"]
    # }

    # exp.create_model()
    # exp.create_trainer()

    # print("\n🚀 Training final model with best params...")
    # exp.train_data()

    # torch.save(exp.model.state_dict(), "best_optuna_model.pt")

    # print("✅ Model saved: best_optuna_model.pt")