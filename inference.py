# -*- coding: utf-8 -*-
"""
Created on Thu May  2 09:41:19 2019

Inference, make predictions and submit

"""
import configs
import argparse
from pathlib import Path
from typing import Callable, List
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import torch
from torch import nn, cuda
from torch.nn import functional as F
import torchvision.models as M
from torch.utils.data import DataLoader
from torch.utils.data import Dataset 
    
import cv2
from PIL import Image
from torchvision.transforms import (
    ToTensor, Normalize, Compose, Resize, CenterCrop, RandomCrop,
    RandomHorizontalFlip, RandomGrayscale)

from dataset import ImetDatasetTTA
from transforms import tensor_transform, train_transform, test_transform, albu_transform, valid_transform
from utils import load_model, set_seed, check_fold 
from senet_models import seresnext101
from train import args



def main():
    set_seed(args.seed)
    
    """
        
    SET PARAMS
        
    """
    ON_KAGGLE = configs.ON_KAGGLE
    N_CLASSES = configs.NUM_CLASSES
    DATA_ROOT = configs.DATA_ROOT
    SIZE      = configs.SIZE
    RUN_ROOT = '../input/seresnext101-folds/' if ON_KAGGLE else '../data/seresnext101/'
    args.debug = True
    use_cuda = cuda.is_available()
    use_sample = args.debug
    num_workers = args.workers
    batch_size = args.batch_size
    train_root = DATA_ROOT / 'train'
    test_root = DATA_ROOT / 'test'
    print('Files present in model directory :', os.listdir(RUN_ROOT))
    print('Files present in data directory  :', os.listdir(DATA_ROOT))
    
    """
        
    LOAD MODEL
        
    """
    model = seresnext101(num_classes=N_CLASSES)      
    if use_cuda:
        model = model.cuda()
        
    def load_model(model: nn.Module, root: str, fold: int, use_cuda: bool):
        """
        Loads model checkpoints
        Choose evaluation mode
        """
        best_model_path = root + 'best-metric_fold'+str(fold)+'.pt'
        if use_cuda:
            state_dict = torch.load(best_model_path)
        else:
            state_dict = torch.load(best_model_path, map_location='cpu')        
        # model’s parameters
        model.load_state_dict(state_dict['model'])
        print('Loaded model from epoch {epoch}, step {step:,}'.format(**state_dict))
        model.eval()
        
    """

    MAKE PREDICTIONS
    
    """
    def get_dataloader(df: pd.DataFrame, image_transform, tta: int) -> DataLoader:
        """
        Calls dataloader to load Imet Dataset with TTA
        """
        return DataLoader(
            ImetDatasetTTA(test_root, df, image_transform, tta),
            shuffle=False,
            batch_size=args.batch_size,
            num_workers=args.workers,
            )
        
def predict(model: nn.Module, root: Path, predict_df: pd.DataFrame, save_root: str,
            image_transform, batch_size: int, tta: int, workers: int, use_cuda: bool):
    """
    Makes preditions
    """    
    
    test_loader = get_dataloader(predict_df, image_transform, tta)
    all_outputs, all_ids = [], []        
    with torch.no_grad():
        for inputs, ids in tqdm(test_loader, desc='Predict'):
            if use_cuda:
                inputs = inputs.cuda()
            outputs = torch.sigmoid(model(inputs))
            all_outputs.append(outputs.data.cpu().numpy())
            all_ids.extend(ids)    
    
    df = pd.DataFrame(
            data=np.concatenate(all_outputs),
            index=all_ids,
            columns=map(str, range(N_CLASSES)))
    df = mean_df(df)
    print('probs: ', df.head(10)) 
    return df    

    predict_kwargs = dict(
            image_transform=test_transform,
            batch_size=batch_size,
            tta=2,
            workers=0,
            use_cuda=use_cuda
            )    

ss = pd.read_csv(DATA_ROOT/'sample_submission.csv')
if use_sample:
    ss = ss.head(100)     
print(ss.head())