# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 16:07:30 2019

@author: Tatiana Gabruseva

"""
import numpy as np
import warnings
import torch
from torch import nn
from catalyst.dl.core import Callback, RunnerState
from sklearn.metrics import fbeta_score, accuracy_score
from sklearn.exceptions import UndefinedMetricWarning


class F2ScoreCallback(Callback):
    def __init__(
        self,
        input_key: str = "targets",
        output_key: str = "logits",          
        num_classes: int = 1
    ):
        self.input_key = input_key
        self.output_key = output_key        
        self.num_classes = num_classes 
        self.metrics = 0  
        
    def reset(self):
        self.metrics = 0            

    def on_loader_start(self, state):
        self.reset()
   
    def get_score(self, y_pred, targets):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=UndefinedMetricWarning)
            return fbeta_score(
                targets, y_pred, beta=2, average='samples')           
              
    def get_metrics(self, probabilities, targets):
        metrics = {}
        argsorted = probabilities.argsort()
        for threshold in [0.05, 0.1, 0.15, 0.2, 0.25]:
            y_pred = binarize_prediction(probabilities, threshold, argsorted)
            metrics[f'valid_f2_th_{threshold:.2f}'] = self.get_score(y_pred, targets)
        print(' | '.join(f'{k} {v:.3f}' for k, v in sorted(
            metrics.items(), key=lambda kv: -kv[1])))  
        return metrics
    
    def on_loader_end(self, state: RunnerState):
        logits: torch.Tensor = state.output[self.output_key].detach().float()
        targets: torch.Tensor = state.input[self.input_key].detach().float()
        probabilities: torch.Tensor = torch.sigmoid(logits)
        metrics = self.get_metrics(probabilities, targets)
        for threshold in [0.05, 0.1, 0.15, 0.2, 0.25]:
            metric_name = f'valid_f2_th_{threshold:.2f}'
            state.metrics.epoch_values[state.loader_name][metric_name] = metrics[metric_name]   
        self.reset()    
                
 
def binarize_prediction(probabilities, num_classes: int, threshold: float, argsorted=None,
                        min_labels=1, max_labels=10):
    """ 
    Return matrix of 0/1 predictions, based in probabilities, 
    the same shape as probabilities    
    Author: Konstantin Lopuhin 
    source: https://github.com/lopuhin/kaggle-imet-2019/tree/master/imet
    """    
    assert probabilities.shape[1] == num_classes
    if argsorted is None:
        argsorted = probabilities.argsort(axis=1)
    max_mask = _make_mask(argsorted, max_labels)
    min_mask = _make_mask(argsorted, min_labels)
    prob_mask = probabilities > threshold
    return (max_mask & prob_mask) | min_mask


def _make_mask(argsorted, top_n: int):
    """
    Author: Konstantin Lopuhin
    source: https://github.com/lopuhin/kaggle-imet-2019/tree/master/imet
    """
    mask = np.zeros_like(argsorted, dtype=np.uint8)
    col_indices = argsorted[:, -top_n:].reshape(-1)
    row_indices = [i // top_n for i in range(len(col_indices))]
    mask[row_indices, col_indices] = 1
    return mask
   
        
        

def validation(
        model: nn.Module, criterion, valid_loader, use_cuda
        ):
    """Model validation
    Calculates f2 score and accuracy score metrics
    """
    model.eval()
    all_predictions, all_targets = [], []
    with torch.no_grad():
        for inputs, targets in valid_loader:
            all_targets.append(targets.numpy().copy())
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)
            predictions = torch.sigmoid(outputs)
            all_predictions.append(predictions.cpu().numpy())
    all_predictions = np.concatenate(all_predictions)
    all_targets = np.concatenate(all_targets)

    def get_score(y_pred):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=UndefinedMetricWarning)
            return fbeta_score(
                all_targets, y_pred, beta=2, average='samples')

    def get_acc_score(y_pred):
        """
        https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html
        """
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=UndefinedMetricWarning)
            return accuracy_score(
                all_targets, y_pred, normalize=True)

    metrics = {}
    argsorted = all_predictions.argsort(axis=1)
    for threshold in [0.05, 0.1, 0.15, 0.2, 0.25]:
        metrics[f'valid_f2_th_{threshold:.2f}'] = get_score(
            binarize_prediction(all_predictions, threshold, argsorted))
    print(' | '.join(f'{k} {v:.3f}' for k, v in sorted(
        metrics.items(), key=lambda kv: -kv[1])))
   
    accuracy = {}
    argsorted = all_predictions.argsort(axis=1)
    for threshold in [0.3, 0.4, 0.5, 0.6, 0.7]:
        accuracy[f'acc_score_{threshold:.2f}'] = get_acc_score(
            binarize_prediction(all_predictions, threshold, argsorted))
    print(' | '.join(f'{k} {v:.3f}' for k, v in sorted(
        accuracy.items(), key=lambda kv: -kv[1])))
    return metrics


__all__ = ["F2ScoreCallback"]        