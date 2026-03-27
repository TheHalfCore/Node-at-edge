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

    # ✅ NEW: data preparation method
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

    def run(self):
        print("Starting experiment...")
        self.load_and_preprocess_data()

        # 👉 Next steps: model, training, etc.

if __name__ == "__main__":
    experiment = UniMiBExperiment(gpu_id=0)
    experiment.run()        