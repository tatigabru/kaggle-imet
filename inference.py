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
    RUN_ROOT = '../input/seresnext101-folds/' if ON_KAGGLE else '../data/results/'
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
        """Loads model checkpoints
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
