import os
import time
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import get_latest_file, iterate_minibatches, check_numpy, process_in_chunks
from .nn_utils import to_one_hot
from collections import OrderedDict
from copy import deepcopy
from tensorboardX import SummaryWriter

from sklearn.metrics import roc_auc_score, log_loss


class Trainer(nn.Module): #a class that encapsulates the training loop, checkpointing, and evaluation of a model
    def __init__(self, model, loss_function, experiment_name=None, warm_start=False, 
                 Optimizer=torch.optim.Adam, optimizer_params={}, verbose=False, 
                 n_last_checkpoints=1, **kwargs):
        """
        :type model: torch.nn.Module (our neural network model is DenseBlock + ODST)
        :param loss_function: the metric to use in trainnig (MSE or CrossEntropy)
        :param experiment_name: a path where all logs and checkpoints are saved
        :param warm_start: when set to True, loads last checpoint
        :param Optimizer: function(parameters) -> optimizer
        :param verbose: when set to True, produces logging information
        """
        super().__init__()
        self.model = model
        self.loss_function = loss_function
        self.verbose = verbose
        self.opt = Optimizer(list(self.model.parameters()), **optimizer_params)
        self.step = 0
        self.n_last_checkpoints = n_last_checkpoints

        if experiment_name is None: #if no experiment name is provided, create one based on the current time
            experiment_name = 'untitled_{}.{:0>2d}.{:0>2d}_{:0>2d}-{:0>2d}'.format(*time.gmtime()[:5])
            if self.verbose:
                print('using automatic experiment name: ' + experiment_name)

        self.experiment_path = os.path.join('logs/', experiment_name)
        if not warm_start and experiment_name != 'debug':
            assert not os.path.exists(self.experiment_path), 'experiment {} already exists'.format(experiment_name)
        self.writer = SummaryWriter(self.experiment_path, comment=experiment_name) #a writer that logs training information to the experiment path, which can be visualized using TensorBoard
        if warm_start: #if warm_start is True, load the last checkpoint from the experiment path
            self.load_checkpoint()
    
    def save_checkpoint(self, tag=None, path=None, mkdir=True, **kwargs): #saves the model and optimizer state to a checkpoint file, which can be loaded later for resuming training or evaluation
        assert tag is None or path is None, "please provide either tag or path or nothing, not both"
        if tag is None and path is None: #if no tag or path is provided, create a temporary checkpoint file with the current step number
            tag = "temp_{}".format(self.step)
        if path is None: #if no path is provided, create a checkpoint file with the given tag in the experiment path
            path = os.path.join(self.experiment_path, "checkpoint_{}.pth".format(tag))
        if mkdir: #create the directory for the checkpoint file if it does not exist
            os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(OrderedDict([
            ('model', self.state_dict(**kwargs)),
            ('opt', self.opt.state_dict()),
            ('step', self.step)
        ]), path)
        if self.verbose:
            print("Saved " + path)
        return path

    def load_checkpoint(self, tag=None, path=None, **kwargs): #loads the model and optimizer state from a checkpoint file, which can be used for resuming training or evaluation
        assert tag is None or path is None, "please provide either tag or path or nothing, not both"
        if tag is None and path is None: #if no tag or path is provided, load the latest temporary checkpoint file from the experiment path
            path = get_latest_file(os.path.join(self.experiment_path, 'checkpoint_temp_[0-9]*.pth'))
        elif tag is not None and path is None: #if a tag is provided but no path, create a checkpoint file path with the given tag in the experiment path
            path = os.path.join(self.experiment_path, "checkpoint_{}.pth".format(tag))
        checkpoint = torch.load(path)

        self.load_state_dict(checkpoint['model'], **kwargs)
        self.opt.load_state_dict(checkpoint['opt'])
        self.step = int(checkpoint['step'])

        if self.verbose:
            print('Loaded ' + path)
        return self

    def average_checkpoints(self, tags=None, paths=None, out_tag='avg', out_path=None): #averages the model parameters from multiple checkpoint files and saves the averaged model to a new checkpoint file, which can be used for evaluation or ensembling
        assert tags is None or paths is None, "please provide either tags or paths or nothing, not both" 
        assert out_tag is not None or out_path is not None, "please provide either out_tag or out_path or both, not nothing"
        if tags is None and paths is None: #if no tags or paths are provided, average the latest temporary checkpoint files from the experiment path, up to n_last_checkpoints
            paths = self.get_latest_checkpoints(
                os.path.join(self.experiment_path, 'checkpoint_temp_[0-9]*.pth'), self.n_last_checkpoints)
        elif tags is not None and paths is None: #if tags are provided but no paths, create checkpoint file paths with the given tags in the experiment path
            paths = [os.path.join(self.experiment_path, 'checkpoint_{}.pth'.format(tag)) for tag in tags]

        checkpoints = [torch.load(path) for path in paths] #load the checkpoint files into a list of dictionaries, where each dictionary contains the model state, optimizer state, and step number
        averaged_ckpt = deepcopy(checkpoints[0]) #create a new checkpoint dictionary by copying the first checkpoint, which will be used to store the averaged model state, optimizer state, and step number
        for key in averaged_ckpt['model']:
            values = [ckpt['model'][key] for ckpt in checkpoints] #for each model parameter key, collect the corresponding parameter values from all checkpoints into a list
            averaged_ckpt['model'][key] = sum(values) / len(values) #average the parameter values by summing them and dividing by the number of checkpoints, and store the averaged value in the new checkpoint dictionary

        if out_path is None:
            out_path = os.path.join(self.experiment_path, 'checkpoint_{}.pth'.format(out_tag))
        torch.save(averaged_ckpt, out_path)

    def get_latest_checkpoints(self, pattern, n_last=None):
        list_of_files = glob.glob(pattern)
        assert len(list_of_files) > 0, "No files found: " + pattern
        return sorted(list_of_files, key=os.path.getctime, reverse=True)[:n_last]

    def remove_old_temp_checkpoints(self, number_ckpts_to_keep=None):
        if number_ckpts_to_keep is None:
            number_ckpts_to_keep = self.n_last_checkpoints
        paths = self.get_latest_checkpoints(os.path.join(self.experiment_path, 'checkpoint_temp_[0-9]*.pth'))
        paths_to_delete = paths[number_ckpts_to_keep:]

        for ckpt in paths_to_delete:
            os.remove(ckpt)

    def train_on_batch(self, *batch, device): #performs a training step on a single batch of data, which includes computing the loss, backpropagating the gradients, and updating the model parameters using the optimizer
        x_batch, y_batch = batch
        x_batch = torch.as_tensor(x_batch, device=device) #convert the input batch to a PyTorch tensor and move it to the specified device (CPU or GPU)
        y_batch = torch.as_tensor(y_batch, device=device)

        self.model.train() #set the model to training mode
        self.opt.zero_grad() #reset the gradients
        loss = self.loss_function(self.model(x_batch), y_batch).mean() #compute the loss
        loss.backward() #backpropagate the gradients
        self.opt.step() #update the model parameters using the optimizer
        self.step += 1
        self.writer.add_scalar('train loss', loss.item(), self.step)
        
        return {'loss': loss}

    def evaluate_classification_error(self, X_test, y_test, device, batch_size=4096): #evaluates the classification error rate of the model on a test dataset, which involves computing the predicted logits for the test inputs, comparing them to the true labels, and calculating the mean error rate
        X_test = torch.as_tensor(X_test, device=device) #convert the test inputs to a PyTorch tensor and move it to the specified device (CPU or GPU)
        y_test = check_numpy(y_test) #convert the test labels to a NumPy array if they are not already, which allows for easier comparison with the predicted logits later on
        self.model.train(False) #set the model to evaluation mode, which disables dropout and other training-specific behaviors
        with torch.no_grad(): #disable gradient computation, which reduces memory usage and speeds up inference during evaluation
            logits = process_in_chunks(self.model, X_test, batch_size=batch_size) 
            logits = check_numpy(logits)
            error_rate = (y_test != np.argmax(logits, axis=1)).mean()
        return error_rate

    def evaluate_mse(self, X_test, y_test, device, batch_size=4096): #evaluates the mean squared error of the model on a test dataset, which involves computing the predicted values for the test inputs, comparing them to the true values, and calculating the mean squared error
        X_test = torch.as_tensor(X_test, device=device)
        y_test = check_numpy(y_test)
        self.model.train(False)
        with torch.no_grad():
            prediction = process_in_chunks(self.model, X_test, batch_size=batch_size)
            prediction = check_numpy(prediction)
            error_rate = ((y_test - prediction) ** 2).mean()
        return error_rate
    
    def evaluate_auc(self, X_test, y_test, device, batch_size=512): #evaluates the area under the ROC curve (AUC) of the model on a test dataset, which involves computing the predicted probabilities for the test inputs, comparing them to the true labels, and calculating the AUC score using the roc_auc_score function from scikit-learn
        X_test = torch.as_tensor(X_test, device=device)
        y_test = check_numpy(y_test)
        self.model.train(False)
        with torch.no_grad():
            logits = F.softmax(process_in_chunks(self.model, X_test, batch_size=batch_size), dim=1)
            logits = check_numpy(logits)
            y_test = torch.tensor(y_test)
            auc = roc_auc_score(check_numpy(to_one_hot(y_test)), logits)
        return auc
    
    def evaluate_logloss(self, X_test, y_test, device, batch_size=512): #evaluates the log loss of the model on a test dataset, which involves computing the predicted probabilities for the test inputs, comparing them to the true labels, and calculating the log loss using the log_loss function from scikit-learn
        X_test = torch.as_tensor(X_test, device=device)
        y_test = check_numpy(y_test)
        self.model.train(False)
        with torch.no_grad():
            logits = F.softmax(process_in_chunks(self.model, X_test, batch_size=batch_size), dim=1)
            logits = check_numpy(logits)
            y_test = torch.tensor(y_test)
            logloss = log_loss(check_numpy(to_one_hot(y_test)), logits)
        return logloss
