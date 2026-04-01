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

from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import LabelEncoder

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
        self.f1_history = []
        self.best_mse = float('inf')
        self.best_f1 = float('-inf')
        self.best_step_mse = 0
        self.best_step_f1 = 0
        self.early_stopping_rounds = 5000
        self.report_frequency = 100
        self.fig = None
        self.axes = None
        self.layer_dim = 64 #number of trees in each NODE layer
        self.num_layers = 8 #umber of layers in the block
        self.tree_dim = 17 #number of output features per tree (default 1, i.e. scalar output)
        self.depth = 6 #depth of each tree (default 6, i.e. 64 leafs per tree)
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

        # Temporarily remove this section to replace with classification setup
        # Compute normalization stats
        # self.mu = self.data.y_train.mean()
        # self.std = self.data.y_train.std()

        # # Normalize targets
        # normalize = lambda x: ((x - self.mu) / self.std).astype(np.float32)

        # self.data.y_train, self.data.y_valid, self.data.y_test = map(
        #     normalize,
        #     [self.data.y_train, self.data.y_valid, self.data.y_test]
        # )

        # print(
        #     "mean = %.5f, std = %.5f" % (self.mu, self.std),
        #     f"in_features = {self.in_features}"
        # )

        # For classification, we need to encode labels and determine the number of classes
        le = LabelEncoder() # Encode string labels to integers if necessary, or just ensure they are in the right format
        self.data.y_train = le.fit_transform(self.data.y_train) # Fit on training labels and transform
        self.data.y_valid = le.transform(self.data.y_valid) # Transform validation labels using the same encoder
        self.data.y_test = le.transform(self.data.y_test) # Transform test labels using the same encoder
        
        self.num_classes = len(le.classes_) # Get the number of unique classes from the label encoder
        print("Labels:", np.unique(self.data.y_train))
        print("Num classes:", self.num_classes)

    def create_model(self):
        dense = lib.DenseBlock(
        self.in_features,
        self.layer_dim,
        num_layers=self.num_layers,
        tree_dim=self.num_classes,
        depth=self.depth,
        flatten_output=False
        ).to(self.device)

        # Proper data-aware init + shape detection
        with torch.no_grad():
            dummy = torch.as_tensor(self.data.X_train[:2048], device=self.device)
            out = dense(dummy)
            flat_dim = out.view(out.size(0), -1).shape[1]

        self.model = nn.Sequential(
            dense,
            #lib.Lambda(lambda x: x.mean(dim=-1).mean(dim=-1)),
            #lib.Lambda(lambda x: x.mean(dim=-1)), # average over trees, keep output shape (batch_size, layer_dim * tree_dim)
            #lib.Lambda(lambda x: x.mean(dim=-2)), # average over layers, keep output shape (batch_size, tree_dim)
            nn.Flatten(), # flatten to (batch_size, layer_dim * tree_dim)
            nn.Linear(flat_dim, self.num_classes) # final linear layer to output logits for each class
        ).to(self.device)
    
        if torch.cuda.device_count() > 1: # Wrap model with DataParallel if multiple GPUs are available
            self.model = nn.DataParallel(self.model) # This will automatically split input across GPUs and gather outputs

    def declare_optimizer_param(self):
        self.optimizer_params = { 'nus':(0.7, 1.0), 'betas':(0.95, 0.998) }
    
    def create_trainer(self):
        weights = compute_class_weight( # Compute class weights to handle class imbalance
        class_weight="balanced", # Automatically adjust weights inversely proportional to class frequencies
        classes=np.unique(self.data.y_train), # Get unique class labels from training data
        y=self.data.y_train # Provide training labels to compute class frequencies and weights
    )
        class_weights = torch.tensor(weights, dtype=torch.float32).to(self.device) # Convert to tensor and move to device
        def weighted_loss(logits, y): # Define weighted cross-entropy loss function
            return F.cross_entropy(logits, y, weight=class_weights) # Use class weights in the loss function
        
        self.trainer = lib.Trainer(
            model=self.model, loss_function=weighted_loss,  # Use weighted cross-entropy loss for classification
            experiment_name=self.experiment_name,
            warm_start=False,
            Optimizer=QHAdam,
            optimizer_params=self.optimizer_params,
            verbose=True,
            n_last_checkpoints=5
        )

    def evaluate_f1(self, X, y, batch_size=4096): # Evaluate F1 score on given data
        self.model.eval() # Set model to evaluation mode
        preds = []

        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                xb = torch.as_tensor(X[i:i+batch_size], device=self.device)
                logits = self.model(xb)

                pred = torch.argmax(logits, dim=1)
                preds.append(pred.cpu().numpy())

        preds = np.concatenate(preds) # Combine predictions from all batches
        return f1_score(y, preds, average="macro")
    
    def train_data(self):
        epochs = 50
        batch_size = 1024
        #batch_size_mse = 16384
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

                #self.mse = self.trainer.evaluate_mse(self.data.X_valid, self.data.y_valid, device=self.device, batch_size=batch_size_mse)
                self.f1 = self.evaluate_f1(self.data.X_valid, self.data.y_valid)
                
                # if self.mse < self.best_mse:
                #     self.best_mse = self.mse
                #     self.best_step_mse = self.trainer.step
                #     self.trainer.save_checkpoint(tag='best_mse')

                # self.mse_history.append(self.mse)
                
                if self.f1 > self.best_f1:
                    self.best_f1 = self.f1
                    self.best_step_f1 = self.trainer.step
                    self.trainer.save_checkpoint(tag='best_f1')
                    
                self.f1_history.append(self.f1)
                
                self.trainer.load_checkpoint()  # last
                self.trainer.remove_old_temp_checkpoints()

                self.update_plots()

                print(f"Step {self.trainer.step}")
                print(f"Loss: {metrics['loss'].item():.5f}")
                #print(f"Val MSE: {self.mse:.5f}")
                print(f"Val F1: {self.f1:.5f}")

            # if self.trainer.step > self.best_step_mse + self.early_stopping_rounds:
            #     print('BREAK. There is no improvement for {} steps'.format(self.early_stopping_rounds))
            #     print("Best step:", self.best_step_mse)
            #     print("Best Val MSE: %0.5f" % self.best_mse)
            #     break
            
            if self.trainer.step > self.best_step_f1 + self.early_stopping_rounds:
                print('BREAK. There is no improvement for {} steps'.format(self.early_stopping_rounds))
                print("Best step:", self.best_step_f1)
                print("Best Val F1: %0.5f" % self.best_f1)
                break
            
    def update_plots(self):
        self.axes[0].clear()
        self.axes[0].plot(self.loss_history)
        self.axes[0].set_title('Loss')
        self.axes[0].grid()

        self.axes[1].clear()
        # self.axes[1].plot(self.mse_history)
        # self.axes[1].set_title('MSE')
        self.axes[1].plot(self.f1_history)
        self.axes[1].set_title('F1 Score')
        self.axes[1].grid()

        plt.pause(0.01)

    def load_checkpoint(self):
        if os.path.exists(self.best_model_path):
            model_to_load = self.model.module if hasattr(self.model, "module") else self.model
            model_to_load.load_state_dict(torch.load(self.best_model_path, map_location=self.device))
            self.model.eval()
        else:
            # self.trainer.load_checkpoint(tag='best_mse')
            self.trainer.load_checkpoint(tag='best_f1')

        # self.mse = self.trainer.evaluate_mse(self.data.X_test, self.data.y_test, device=self.device)
        self.f1 = self.evaluate_f1(self.data.X_test, self.data.y_test)
        print('Best step: ', self.trainer.step)
        # print("Test MSE: %0.5f" % (self.mse))
        print("Test F1: %0.5f" % (self.f1))

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