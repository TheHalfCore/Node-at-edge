import os
import sys
import time

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

# Optional: if lib is your local module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import lib

from qhoptim.pyt import QHAdam
from tqdm import tqdm
from IPython.display import clear_output

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(BASE_DIR, "..", "notebooks", "data")
DATA_PATH = os.path.abspath(DATA_PATH)


class UniMiBExperiment:
    def __init__(self, gpu_id=0):
        self.gpu_id = gpu_id
        self.device = self._setup_device()
        self.experiment_name = self._create_experiment_name()

        # Will be set later
        self.data = None
        self.in_features = None
        self.mu = None
        self.std = None
        self.model = None
        self.optimizer_params = None
        self.trainer = None
        self.loss_history = []
        self.mse_history = []
        self.best_mse = float('inf')
        self.best_step_mse = 0
        self.early_stopping_rounds = 5000
        self.report_frequency = 100
        self.fig = None
        self.axes = None
        self.layer_dim = 8 #number of output features per tree in each layer
        self.num_layers = 4 #umber of layers in the block
        self.tree_dim = 1 #number of output features per tree (default 1, i.e. scalar output)
        self.depth = 6
        self.best_model_path = "best_model.pt"

    def _setup_device(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        return device

    def _create_experiment_name(self):
        name = "UniMiB_data"
        timestamp = time.gmtime()
        experiment_name = "{}_{}.{:0>2d}.{:0>2d}_{:0>2d}-{:0>2d}".format(
            name, *timestamp[:5]
        )
        print("experiment:", experiment_name)
        return experiment_name

    def load_and_preprocess_data(self):
        self.data = lib.Dataset(
            "UniMiB",
            data_path=DATA_PATH,
            random_state=1337,
            quantile_transform=True,
            quantile_noise=1e-3
        )

        self.in_features = self.data.X_train.shape[1]

        # Compute normalization stats
        self.mu = self.data.y_train.mean()
        self.std = self.data.y_train.std()

        # Normalize targets
        normalize = lambda x: ((x - self.mu) / self.std).astype(np.float32)

        self.data.y_train, self.data.y_valid, self.data.y_test = map(
            normalize,
            [self.data.y_train, self.data.y_valid, self.data.y_test]
        )

        print(
            "mean = %.5f, std = %.5f" % (self.mu, self.std),
            f"in_features = {self.in_features}"
        )

    def create_model(self):
        self.model = nn.Sequential(
            lib.DenseBlock(self.in_features, self.layer_dim, num_layers=self.num_layers
                           , tree_dim=self.tree_dim, depth=self.depth, flatten_output=False),
            lib.Lambda(lambda x: x.mean(dim=-1).mean(dim=-1)),
            #lib.Lambda(lambda x: x.mean(dim=-1)),
        ).to(self.device)

        with torch.no_grad():
            res = self.model(torch.as_tensor(self.data.X_train[:128], device=self.device))
            # trigger data-aware init
    
        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)

    def declare_optimizer_param(self):
        self.optimizer_params = { 'nus':(0.7, 1.0), 'betas':(0.95, 0.998) }
    
    def create_trainer(self):
        self.trainer = lib.Trainer(
            model=self.model, loss_function=F.mse_loss,
            experiment_name=self.experiment_name,
            warm_start=False,
            Optimizer=QHAdam,
            optimizer_params=self.optimizer_params,
            verbose=True,
            n_last_checkpoints=5
        )

    def train_data(self):
        epochs = 10
        batch_size = 1024
        batch_size_mse = 16384
        report_frequency = self.report_frequency
        for batch in lib.iterate_minibatches(self.data.X_train, self.data.y_train, batch_size=batch_size, shuffle=True, epochs=epochs):
            print("Number of trainer.step:", self.trainer.step)
            metrics = self.trainer.train_on_batch(*batch, device=self.device)
            
            # FIX: convert tensor to number
            self.loss_history.append(metrics['loss'].item())
            if self.trainer.step < report_frequency:
                report_frequency = self.trainer.step
            else:
                report_frequency = self.report_frequency

            if self.trainer.step % report_frequency == 0:
                self.trainer.save_checkpoint()
                self.trainer.average_checkpoints(out_tag='avg')
                self.trainer.load_checkpoint(tag='avg')

                self.mse = self.trainer.evaluate_mse(self.data.X_valid, self.data.y_valid, device=self.device, batch_size=batch_size_mse)

                if self.mse < self.best_mse:
                    self.best_mse = self.mse
                    self.best_step_mse = self.trainer.step
                    self.trainer.save_checkpoint(tag='best_mse')

                self.mse_history.append(self.mse)
                
                self.trainer.load_checkpoint()  # last
                self.trainer.remove_old_temp_checkpoints()

                self.update_plots()

                print(f"Step {self.trainer.step}")
                print(f"Loss: {metrics['loss'].item():.5f}")
                print(f"Val MSE: {self.mse:.5f}")

            if self.trainer.step > self.best_step_mse + self.early_stopping_rounds:
                print('BREAK. There is no improvement for {} steps'.format(self.early_stopping_rounds))
                print("Best step:", self.best_step_mse)
                print("Best Val MSE: %0.5f" % self.best_mse)
                break
    def update_plots(self):
        self.axes[0].clear()
        self.axes[0].plot(self.loss_history)
        self.axes[0].set_title('Loss')
        self.axes[0].grid()

        self.axes[1].clear()
        self.axes[1].plot(self.mse_history)
        self.axes[1].set_title('MSE')
        self.axes[1].grid()

        plt.pause(0.01)

    def load_checkpoint(self):
        if os.path.exists(self.best_model_path):
            model_to_load = self.model.module if hasattr(self.model, "module") else self.model
            model_to_load.load_state_dict(torch.load(self.best_model_path, map_location=self.device))
            self.model.eval()
        else:
            self.trainer.load_checkpoint(tag='best_mse')

        self.mse = self.trainer.evaluate_mse(self.data.X_test, self.data.y_test, device=self.device)
        print('Best step: ', self.trainer.step)
        print("Test MSE: %0.5f" % (self.mse))

    def run(self):
        print("Starting experiment...")
        self.load_and_preprocess_data()

        print("Create Model...")
        self.create_model()

        print("declare optimizer param...")
        self.declare_optimizer_param()

        print("create trainer...")
        self.create_trainer()

        if os.path.exists(self.best_model_path):
            print("Model loaded. Skipping training.")
        else:
            print("Start trainning...")
            plt.ion()
            self.fig, self.axes = plt.subplots(1, 2, figsize=(18, 6))
            self.train_data()
            torch.save(self.model.state_dict(), self.best_model_path)
            # Keep final plot visible
            plt.ioff()     # turn OFF interactive mode
            plt.show()     # now it BLOCKS and stays open
            print("Trainning end")
        

        print("Load checkpoint...")
        self.load_checkpoint()
        print("The end")


if __name__ == "__main__":
    experiment = UniMiBExperiment(gpu_id=0)
    experiment.run()        