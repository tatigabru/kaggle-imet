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
from sklearn.metrics import fbeta_score, accuracy_score
from sklearn.exceptions import UndefinedMetricWarning
from collections import OrderedDict

import torch
from torch import nn, cuda
from torch.optim import Adam, lr_scheduler
from torch.utils.data import DataLoader
from datetime import datetime

from dataset import ImetDataset
from transforms import tensor_transform, train_transform, test_transform, albu_transform, valid_transform
from utils import load_model, set_seed, check_fold 
from senet_models import seresnext101

from catalyst.dl.runner import SupervisedRunner
from catalyst.dl.callbacks import InferCallback
from catalyst.dl.callbacks import EarlyStoppingCallback, F1ScoreCallback, CheckpointCallback
from callbacks import F2ScoreCallback


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
    arg('--learning-rate', type=float, default=1e-3, help='Initial learning rate')
    arg('--step', type=int, default=1, help='Current training step')
    arg('--patience', type=int, default=4)    
    arg('--criterion', type=str, default='bce', help='Criterion')
    arg('--optimizer', default='Adam', help='Name of the optimizer')    
    arg('--continue_train', type=bool, default=False)   
    arg('--checkpoint', type=str, default=Path('../results'), help='Checkpoints root path')
    arg('--workers', type=int, default=2)   
    arg('--debug', type=bool, default=True)
    args = parser.parse_args() 
    
    set_seed(args.seed)
    
    """
    
    SET PARAMS
    
    """
    args.debug = True
    ON_KAGGLE = configs.ON_KAGGLE
    N_CLASSES = configs.NUM_CLASSES
    args.image_size = configs.SIZE
    args.data_root = configs.DATA_ROOT
    use_cuda = cuda.is_available()
    fold = args.fold
    num_workers = args.workers
    num_epochs = args.n_epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate
                  
    """

    LOAD DATA
    
    """
    print(os.listdir(args.data_root))
    folds = pd.read_csv(args.data_root / 'folds.csv')     
    train_root = args.data_root / 'train'   
        
    if args.debug:    
        folds = folds.head(50)        
    train_fold = folds[folds['fold'] != fold]
    valid_fold = folds[folds['fold'] == fold] 
    check_fold(train_fold, valid_fold)          
    
        
    def get_dataloader(df: pd.DataFrame, image_transform) -> DataLoader:
        """
        Calls dataloader to load Imet Dataset
        """
        return DataLoader(
            ImetDataset(train_root, df, image_transform),
            shuffle=True,
            batch_size=batch_size,
            num_workers=num_workers,
            )         
    
    train_loader = get_dataloader(train_fold, image_transform=train_transform)
    valid_loader = get_dataloader(valid_fold, image_transform=test_transform)
    print('{} items in train, {} in valid'.format(len(train_loader.dataset),len(valid_loader.dataset)))   
    loaders = OrderedDict()
    loaders["train"] = train_loader
    loaders["valid"] = valid_loader    
    
    """
    
    MODEL
    
    """
    model = seresnext101(num_classes=N_CLASSES)      
    if use_cuda:
        model = model.cuda()

    criterion = nn.BCEWithLogitsLoss() 
    optimizer = Adam(model.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=args.patience)
    
    """
    
    MODEL RUNNER
    
    """
    # call an instance of the model runner
    runner = SupervisedRunner()
    
    # logs folder
    current_time = datetime.now().strftime('%b%d_%H_%M')
    prefix = f'{current_time}_{args.model_name}'
    logdir = os.path.join(args.run_root, prefix)
    os.makedirs(logdir, exist_ok=False) 
       
    print('Train session    :', prefix)  
    print('\tOn KAGGLE      :', ON_KAGGLE)
    print('\tDebug          :', args.debug)
    print('\tClasses nember :', N_CLASSES)
    print('\tModel          :', args.model_name)
    print('\tParameters     :', model.parameters())
    print('\tImage size     :', args.image_size)
    print('\tEpochs         :', num_epochs)
    print('\tWorkers        :', num_workers)
    print('\tLog dir        :', logdir)    
    print('\tLearning rate  :', learning_rate)
    print('\tBatch size     :', batch_size)
    print('\tPatience       :', args.patience)
    
    if args.continue_train:
        state = load_model(model, args.checkpoint)
        epoch = state['epoch']
        step = state['step']
        print('Loaded model weights from {}, epoch {}, step {}'.format(args.checkpoint, epoch, step))            
          
    # model training
    runner.train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        loaders=loaders,
        callbacks=[        
        F1ScoreCallback(threshold=0.5),
        EarlyStoppingCallback(patience=args.patience, min_delta=0.01)
        ],
        logdir=logdir,
        num_epochs=num_epochs,
        verbose=True        
    )    
    
    # by default it only plots loss, works in IPython Notebooks    
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
    


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = cuda.is_available()
    main()
