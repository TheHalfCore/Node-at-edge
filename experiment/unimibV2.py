import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import time
import uuid
import shutil
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
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
    def __init__(self, gpu_id=0, layer_dim = 64, num_layers = 8, depth = 6
                 , is_generate_graph = True, epochs = 50, batch_size = 512, is_cpu = False, delete_logs=True):
        self.gpu_id = gpu_id
        self.is_cpu = is_cpu
        self.batch_size = batch_size
        self.device = self._setup_device()
        self.experiment_name = self._create_experiment_name()
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
        self.report_frequency = 200
        self.fig = None
        self.axes = None
        self.layer_dim = layer_dim
        self.num_layers = num_layers
        # self.tree_dim = 8
        self.depth = depth
        self.best_model_path = f"best_model_ld-{self.layer_dim}_nl-{self.num_layers}.pt"
        self.choice_function = lib.entmax15
        self.bin_function = lib.entmoid15
        self.is_generate_graph = is_generate_graph
        self.epochs = epochs
        self.delete_logs = self.delete_logs
        self.gpu_allocated_KB = None
        self.gpu_reserved_KB = None
        self.cpu_activation_memory_KB = None
        self.cpu_estimate_inference_size_KB = None
        self.total_params = None
        self.total_params_size_with_buffer_KB = None

    def _setup_device(self):
        device = "cuda" if torch.cuda.is_available() and self.is_cpu == False else "cpu"
        print(f"Using device: {device}")
        return device

    def _create_experiment_name(self):
        if self.delete_logs == True:
            self.delete_logs()

        name = "UniMiB_data"
        experiment_name = f"{name}_{uuid.uuid4().hex[:8]}"
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

        # For classification, we need to encode labels and determine the number of classes
        le = LabelEncoder() # Encode string labels to integers if necessary, or just ensure they are in the right format
        self.data.y_train = le.fit_transform(self.data.y_train) # Fit on training labels and transform
        self.data.y_valid = le.transform(self.data.y_valid) # Transform validation labels using the same encoder
        self.data.y_test = le.transform(self.data.y_test) # Transform test labels using the same encoder
        
        self.num_classes = len(le.classes_) # Get the number of unique classes from the label encoder
        print("Num classes:", self.num_classes)

        print("Train samples:", len(self.data.X_train))
        print("Validation samples:", len(self.data.X_valid))
        print("Test samples:", len(self.data.X_test))

        total = len(self.data.X_train) + len(self.data.X_valid) + len(self.data.X_test)
        print("Total samples:", total)
        # n_samples = len(self.data.X_train)
        # reommended_batch_size = min(512, max(32, n_samples // 20))
        # print(f"reommended_batch_size: {reommended_batch_size}")

    def create_model(self):
        dense = lib.DenseBlock(
        self.in_features,
        self.layer_dim,
        num_layers=self.num_layers,
        tree_dim=self.num_classes + 1,
        depth=self.depth,
        flatten_output=False,
        choice_function=self.choice_function, 
        bin_function=self.bin_function
        ).to(self.device)

        # Proper data-aware init + shape detection
        with torch.no_grad():
            dummy = torch.as_tensor(self.data.X_train[:2048], device=self.device)
            out = dense(dummy)
            flat_dim = out.view(out.size(0), -1).shape[1]

        self.model = nn.Sequential(
            dense,
            # lib.Lambda(lambda x: x[..., :self.num_classes].mean(dim=-2)),
            nn.Flatten(), # flatten to (batch_size, layer_dim * tree_dim)
            nn.Linear(flat_dim, self.num_classes) # final linear layer to output logits for each class
        ).to(self.device)

        with torch.no_grad():
            res = self.model(torch.as_tensor(self.data.X_train[:2000], device=self.device))
            # trigger data-aware init
    
        if torch.cuda.device_count() > 1: # Wrap model with DataParallel if multiple GPUs are available
            self.model = nn.DataParallel(self.model) # This will automatically split input across GPUs and gather outputs
        
        self.estimate_model_size_with_buffer()
        if self.is_cpu == True: 
            self.measure_cpu_estimate_inference_memory()
        else:
            self.measure_gpu_estimate_inference_memory()

    def measure_gpu_estimate_inference_memory(self):
        self.model.eval()
        x = torch.randn(1, self.in_features).to("cuda")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        with torch.no_grad():
            _ = self.model(x)

        self.gpu_allocated_KB = torch.cuda.max_memory_allocated() / (1024)
        self.gpu_reserved_KB = torch.cuda.max_memory_reserved() / (1024)

    def measure_cpu_estimate_inference_memory(self):
        self.model.eval()
        x = torch.randn(1, self.in_features)  # batch size = 1 (edge case)
        total_bytes = 0
        def hook_fn(module, input, output):
            nonlocal total_bytes
            if isinstance(output, torch.Tensor):
                total_bytes += output.nelement() * output.element_size()

        hooks = []
        for layer in self.model.modules():
            hooks.append(layer.register_forward_hook(hook_fn))

        with torch.no_grad():
            _ = self.model(x)

        for h in hooks:
            h.remove()

        self.cpu_activation_memory_KB = total_bytes / (1024)

        # Add model size
        param_bytes = sum(p.nelement() * p.element_size() for p in self.model.parameters())
        self.cpu_estimate_inference_size_KB = (param_bytes + total_bytes) / (1024)


    def estimate_model_size_with_buffer(self):
        param_size = 0
        for param in self.model.parameters():
            param_size += param.nelement() * param.element_size()

        buffer_size = 0
        for buffer in self.model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        self.total_params = sum(p.numel() for p in self.model.parameters())
        self.total_params_size_with_buffer_KB = (param_size + buffer_size) / (1024)

    def declare_optimizer_param(self):
        self.optimizer_params = { 'nus':(0.7, 1.0), 'betas':(0.95, 0.998), 'lr': 7.657337189649465e-05 }
    
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
            model=self.model, 
            loss_function=weighted_loss,
            #loss_function=F.cross_entropy,
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
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        epochs = self.epochs
        report_frequency = self.report_frequency

        for batch in lib.iterate_minibatches(self.data.X_train, self.data.y_train, batch_size=self.batch_size, shuffle=True, epochs=epochs):
            # print("Number of trainer.step:", self.trainer.step)
            metrics = self.trainer.train_on_batch(*batch, device=self.device)
            
            self.loss_history.append(metrics['loss'].item())

            if self.trainer.step % report_frequency == 0:
                self.trainer.save_checkpoint()
                self.trainer.average_checkpoints(out_tag='avg')
                self.trainer.load_checkpoint(tag='avg')

                self.f1 = self.evaluate_f1(self.data.X_valid, self.data.y_valid, self.batch_size)
            
                
                if self.f1 > self.best_f1:
                    self.best_f1 = self.f1
                    self.best_step_f1 = self.trainer.step
                    self.trainer.save_checkpoint(tag='best_f1')
                    
                self.f1_history.append(self.f1)
                
                self.trainer.load_checkpoint()  # last
                self.trainer.remove_old_temp_checkpoints()
                
                if self.is_generate_graph:
                    self.update_plots()

                print(f"Step {self.trainer.step}")
                print(f"Loss: {metrics['loss'].item():.5f}")
                print(f"Val F1: {self.f1:.5f}")
          
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
            self.trainer.load_checkpoint(tag='best_f1')

        self.f1 = self.evaluate_f1(self.data.X_test, self.data.y_test)
        print('Best step: ', self.trainer.step)
        print("Test F1: %0.5f" % (self.f1))

    def delete_logs(self):
        LOG_ROOT = os.path.join(os.getcwd(), "logs")
        if os.path.exists(LOG_ROOT):
            shutil.rmtree(LOG_ROOT)
            os.makedirs(LOG_ROOT)
            print("Delete logs done..")

    def run(self):
        print("Starting experiment...")
        self.load_and_preprocess_data()

        print("Create Model...")
        self.create_model()

        print("declare optimizer param...")
        self.declare_optimizer_param()
        print(f"total_params: {self.total_params}, total_params_size_with_buffer_KB: {self.total_params_size_with_buffer_KB}, cpu_activation_memory_KB: {self.cpu_activation_memory_KB}, cpu_estimate_inference_size_KB: {self.cpu_estimate_inference_size_KB}")

        print("create trainer...")
        self.create_trainer()

        if os.path.exists(self.best_model_path):
            print("Model loaded. Skipping training.")
        else:
            print("Start trainning...")
            if is_generate_graph:
                plt.ion()
                self.fig, self.axes = plt.subplots(1, 2, figsize=(18, 6))
            self.train_data()
            torch.save(self.model.state_dict(), self.best_model_path)
            if self.is_generate_graph:
                print("Close the graph to end")
                # Keep final plot visible
                plt.ioff()     # turn OFF interactive mode
                plt.show()     # now it BLOCKS and stays open
            print("Trainning end")
        

        print("Load checkpoint...")
        self.load_checkpoint()
        # print("Delete logs...")
        # self.delete_logs()
        print("The end")

if __name__ == "__main__":
    layer_dim = 16
    num_layers = 3
    depth = 3
    epochs = 145
    is_generate_graph = False
    batch_size = 96
    is_cpu = True
    experiment = UniMiBExperiment(gpu_id=0, 
                                  layer_dim=layer_dim, 
                                  num_layers=num_layers, 
                                  depth=depth, 
                                  is_generate_graph=is_generate_graph,
                                  batch_size=batch_size,
                                  is_cpu=is_cpu,
                                   epochs=epochs )
    experiment.run()