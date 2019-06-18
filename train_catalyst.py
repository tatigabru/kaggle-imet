# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 02:40:23 2019

@author: New
"""
import configs
import argparse
import os
from pathlib import Path
import warnings
from typing import Dict
import numpy as np
import pandas as pd
from sklearn.metrics import fbeta_score
from sklearn.exceptions import UndefinedMetricWarning
import tqdm
from collections import Counter, OrderedDict

import torch
from torch import nn, cuda
from torch.optim import Adam, lr_scheduler
from torch.backends import cudnn
from torch.nn import functional as F
import torchvision.models as M
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from  sklearn.metrics import accuracy_score
from datetime import datetime

from my_dataset import TrainDataset, TestDataset 
from transforms import tensor_transform, train_transform, test_transform, albu_transform, valid_transform
from utils import load_model, set_seed, check_fold 
from senet_models import seresnext101

import albumentations as A
from catalyst.dl import utils
from catalyst.dl.runner import SupervisedRunner
from catalyst.dl.callbacks import InferCallback
#from catalyst.dl.callbacks import UtilsFactory
from catalyst.dl.callbacks import EarlyStoppingCallback, F1ScoreCallback, CheckpointCallback



def main():
    
    parser = argparse.ArgumentParser()
    arg = parser.add_argument   
    arg('--seed', type=int, default=1234, help='Random seed')
    arg('--model-name', type=str, default=Path('seresnext101'), help='String model name used for saving')
    arg('--run-root', type=Path, default=Path('../results'), help='Directory for saving model')
    arg('--data-root', type=Path, default=Path('../data'))
    arg('--image-size', type=int, default=224, help='Image size for training')
    arg('--batch-size', type=int, default=16, help='Batch size during training')
    arg('--fold', type=int, default=0, help='Validation fold')
    arg('--n-epochs', type=int, default=10, help='Epoch to run')
    arg('--lr', type=float, default=1e-3, help='Initial learning rate')
    arg('--step', type=int, default=1, help='Current training step')
    arg('--patience', type=int, default=4)    
    arg('--criterion', type=str, default='bce', help='Criterion')
    arg('--optimizer', default='Adam', help='Name of the optimizer')    
    arg('--continue_train', type=bool, default=False)   
    arg('--checkpoint', type=str, default=None, help='Checkpoint file path')
    arg('--workers', type=int, default=2)   
    arg('--debug', type=bool, default=True)
    args = parser.parse_args() 
    args.debug = False
    set_seed(args.seed)  
            
    """

    LOAD DATA
    
    """
    print(os.listdir(args.data_root))
    folds = pd.read_csv(args.data_root / 'folds.csv')     
    fold = args.fold
    train_root = args.data_root / 'train'    
        
    if args.debug:    
        folds = folds.head(50)
        
    train_fold = folds[folds['fold'] != fold]
    valid_fold = folds[folds['fold'] == fold] 
                   
    N_CLASSES = configs.NUM_CLASSES
    
    
    def get_dataloader(df: pd.DataFrame, image_transform) -> DataLoader:
        """
        Calls dataloader to load Imet Dataset
        """
        return DataLoader(
            TrainDataset(train_root, df, image_transform),
            shuffle=True,
            batch_size=args.batch_size,
            num_workers=args.workers,
            )         
    
    train_loader = get_dataloader(train_fold, image_transform=train_transform)
    valid_loader = get_dataloader(valid_fold, image_transform=test_transform)
    print(f'{len(train_loader.dataset):,} items in train, '
          f'{len(valid_loader.dataset):,} in valid')
    
    loaders = OrderedDict()
    loaders["train"] = train_loader
    loaders["valid"] = valid_loader    
    
    """
    
    MODEL
    
    """
    model = seresnext101(num_classes=N_CLASSES)      
    use_cuda = cuda.is_available()
    if use_cuda:
        model = model.cuda()

    criterion = nn.BCEWithLogitsLoss() 
    optimizer = Adam(model.parameters())
    scheduler = lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)
    
    """
    
    MODEL RUNNER
    
    """
    # call instance of the model runner
    runner = SupervisedRunner()
    
    # experiment setup
    current_time = datetime.now().strftime('%b%d_%H_%M')
    prefix = f'{current_time}_{args.model_name}'
    logdir = os.path.join(args.run_root, prefix)
    os.makedirs(logdir, exist_ok=False) 
    num_epochs = args.n_epochs

    if args.continue_train:
        checkpoint = UtilsFactory.load_checkpoint(args.checkpoint)
        UtilsFactory.unpack_checkpoint(checkpoint, model=model)
        checkpoint_epoch = checkpoint['epoch']
        print('Loaded model weights from', args.checkpoint)
        print('Epoch   :', checkpoint_epoch)
        print('Metrics:', checkpoint['epoch_metrics'])

    
          
    # model training
    runner.train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        loaders=loaders,
        callbacks=[        
        F1ScoreCallback(threshold=0.2),
        EarlyStoppingCallback(patience=args.patience, min_delta=0.01)
        ],
        logdir=logdir,
        num_epochs=num_epochs,
        verbose=True        
    )    
    
    # by default it only plots loss    
    #utils.plot_metrics(logdir=logdir, metrics=["loss", "_base/lr"])  

    loaders = OrderedDict([("infer", loaders["train"])])
    runner.infer(
        model=model,
        loaders=loaders,
        callbacks=[
        CheckpointCallback(
            resume=f"{logdir}/checkpoints/best.pth"),
        InferCallback()
    ],
    )
    print(runner.callbacks[1].predictions["logits"])
    


def validation(
        model: nn.Module, criterion, valid_loader, use_cuda,
        ) -> Dict[str, float]:
    """Model validation
    Calculates f2 score and accuracy score metrics
    """
    model.eval()
    all_losses, all_predictions, all_targets = [], [], []
    with torch.no_grad():
        for inputs, targets in valid_loader:
            all_targets.append(targets.numpy().copy())
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            all_losses.append(_reduce_loss(loss).item())
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
    metrics['valid_loss'] = np.mean(all_losses)
    print(' | '.join(f'{k} {v:.3f}' for k, v in sorted(
        metrics.items(), key=lambda kv: -kv[1])))
   
    accuracy = {}
    argsorted = all_predictions.argsort(axis=1)
    for threshold in [0.3, 0.4, 0.5, 0.6, 0.7]:
        accuracy[f'acc_score_{threshold:.2f}'] = get_acc_score(
            binarize_prediction(all_predictions, threshold, argsorted))
    accuracy['valid_loss'] = np.mean(all_losses)
    print(' | '.join(f'{k} {v:.3f}' for k, v in sorted(
        accuracy.items(), key=lambda kv: -kv[1])))
    return metrics


def binarize_prediction(probabilities, threshold: float, argsorted=None,
                        min_labels=1, max_labels=10):
    """ 
    Return matrix of 0/1 predictions, based in probabilities, 
    same shape as probabilities    
    """
    N_CLASSES = configs.NUM_CLASSES
    assert probabilities.shape[1] == N_CLASSES
    if argsorted is None:
        argsorted = probabilities.argsort(axis=1)
    max_mask = _make_mask(argsorted, max_labels)
    min_mask = _make_mask(argsorted, min_labels)
    prob_mask = probabilities > threshold
    return (max_mask & prob_mask) | min_mask


def _make_mask(argsorted, top_n: int):
    mask = np.zeros_like(argsorted, dtype=np.uint8)
    col_indices = argsorted[:, -top_n:].reshape(-1)
    row_indices = [i // top_n for i in range(len(col_indices))]
    mask[row_indices, col_indices] = 1
    return mask


def _reduce_loss(loss):
    return loss.sum() / loss.shape[0]  


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = cuda.is_available()
    main()
