#The following code is an adaptation of the source code presented in https://github.com/Bjarten/early-stopping-pytorch

import numpy as np
import torch
import copy

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, trace_func=print, es_thr = 80):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
                            
            es_thr (int) : The epoch threshold for activation of early stopping. After the threshold
                            is exceeded, early stopping starts to check loss values.
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss_value = None # The best score (the lowest loss value)  achieved so far in early stop period
        self.best_auroc_score = None # The best auroc score that has been calculated so far during the training
        self.early_stop = False # Boolean trigger for stopping training
        self.val_loss_min = np.Inf
        self.delta = delta
        self.best_model = None # the model that achieved the best auroc score
        
        self.best_loss_epoch = None # the number of epochs that has been waited to get the best loss value
        self.best_auroc_epoch = None # the number of epochs that has been waited to get the best auroc score

        self.trace_func = trace_func
        
        self.es_thr = es_thr # Threshold for number of epochs that should be waited before starting early stop checks
        self.thr_exceeded = False #Boolean trigger for checking if the early stop threshold is exceeded
        self.epoch_count = 1
    def __call__(self, val_loss, val_auroc, model):
        """
        args:
        val_loss = validation loss
        val_auroc = auroc score calculated on validation data
        model = model that achieved the provided scores
        """
        

        score = -val_loss
        #first time defining best score and model 
        if self.best_loss_value is None:
            self.best_loss_value = score
            self.best_auroc_score = val_auroc
            self.best_model = copy.deepcopy(model)
            self.best_loss_epoch = self.epoch_count
            self.best_auroc_epoch = self.epoch_count
            
        #when the new loss value is not better than the best one observed before
        elif (score < self.best_loss_value + self.delta) and self.thr_exceeded:
            self.counter += 1
            #self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}. Best score was {-self.best_score}')
            if self.counter >= self.patience: #If the loss didn't decrease enough for specified number of patience epochs, training is stopped by early stop trigger
                self.early_stop = True
        #when the new loss value is better than the best one observed before
        elif not (score < self.best_loss_value + self.delta):
            self.best_loss_value = score
            self.best_loss_epoch = self.epoch_count
            #The new best loss value resets early stop counter
            self.counter = 0
        #Checks if it is time to start early stop observations after waiting enough amount of epochs specified with es_thr
        if self.epoch_count >= self.es_thr:
           self.thr_exceeded = True
           
        #Checks if the provided auroc score is better then the best score that has been observed so far
        if val_auroc > self.best_auroc_score:
            self.best_auroc_score = val_auroc
            self.best_model = copy.deepcopy(model)
            self.best_auroc_epoch = self.epoch_count
            
           
        
        self.epoch_count += 1

