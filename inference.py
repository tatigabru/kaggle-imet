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
from collections import Counter, OrderedDict
    
import cv2
from PIL import Image
from torchvision.transforms import (
    ToTensor, Normalize, Compose, Resize, CenterCrop, RandomCrop,
    RandomHorizontalFlip, RandomGrayscale)

from dataset import ImetDatasetTTA
from transforms import tensor_transform, albu_transform, valid_transform
from utils import load_model, set_seed, check_fold 
from senet_models import seresnext101

from catalyst.dl.callbacks import InferCallback, CheckpointCallback
from catalyst.dl.runner import SupervisedRunner



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





def main():
    
    parser = argparse.ArgumentParser()
    arg = parser.add_argument   
    arg('--seed', type=int, default=1234, help='Random seed')
    arg('--model-name', type=str, default=Path('seresnext101'), help='String model name used for saving')
    arg('--run-root', type=Path, default=Path('../results'), help='Directory for saving model')
    arg('--data-root', type=Path, default=Path('../data'))
    arg('--batch-size', type=int, default=16, help='Batch size during training')      
    arg('--checkpoint', type=str, default=Path('../results'), help='Checkpoint file path')
    arg('--workers', type=int, default=2)   
    arg('--debug', type=bool, default=True)
    args = parser.parse_args()  

    set_seed(args.seed)
    
    """
        
    SET PARAMS
        
    """
    ON_KAGGLE = configs.ON_KAGGLE
    N_CLASSES = configs.NUM_CLASSES
    DATA_ROOT = configs.DATA_ROOT
    args.checkpoint = '../input/seresnext101-folds/' if ON_KAGGLE else '../seresnext101/'
    args.debug = False
    use_cuda = cuda.is_available()
    use_sample = args.debug
    num_workers = args.workers
    batch_size = args.batch_size
    test_root = DATA_ROOT / 'test'
    print('Files present in model directory :', os.listdir(args.checkpoint))
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
        best_model_path = root+'best-metric_fold'+str(fold)+'.pt'
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
            batch_size=batch_size,
            num_workers=num_workers,
            )
        
    def predict(model: nn.Module, predict_df: pd.DataFrame, 
                image_transform, tta: int, use_cuda: bool) -> pd.DataFrame:
        """
        Makes predictions
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
        df = df.groupby(level=0).mean()  
        print('probs: ', df.head(10)) 
        return df  

    
    def predict_catalyst(df, fold, image_transform, tta):
        # catalyst way
        # call instance of the model runner
        runner = SupervisedRunner()
        
        test_loader = get_dataloader(df, image_transform, tta)
        loaders = OrderedDict()
        loaders["test"] = test_loader    
        loaders = OrderedDict([("infer", loaders["test"])])
        runner.infer(
                model=model,
                loaders=loaders,
                callbacks=[
                CheckpointCallback(
                    resume=args.checkpoint+'best-metric_fold'+str(fold)+'.pt'),
                InferCallback()
                ],
            )
        preds=runner.callbacks[1].predictions["logits"]         
        print(preds)
        return preds
    
    """
    
    MAKE SUBMISSION
    
    """  
    ss = pd.read_csv(DATA_ROOT/'sample_submission.csv')
    if use_sample:
        ss = ss.head(100)     
    print(ss.head())

    sample_submission = pd.read_csv(
        DATA_ROOT / 'sample_submission.csv', index_col='id')
        
    def get_classes(item):
        return ' '.join(cls for cls, is_present in item.items() if is_present)
    
    predict_kwargs = dict(
            image_transform=valid_transform,            
            tta=2,
            use_cuda=use_cuda
            )   
    
    dfs = []
    folds = [0, 11, 2, 3, 4]
    for fold in folds:
        load_model(model, args.checkpoint, fold, use_cuda)
        df = predict(model, ss, **predict_kwargs)
        df = df.reindex(sample_submission.index)
        dfs.append(df)
        print(dfs)            
    df = pd.concat(dfs)
    print(df.head())
    # average 5 folds
    df = df.groupby(level=0).mean() 
    df[:] = binarize_prediction(df.values, num_classes=N_CLASSES, threshold=0.11)
    df = df.apply(get_classes, axis=1)
    df.name = 'attribute_ids'
    df.to_csv('submission.csv', header=True)
    print(df.head()) 

    all_ids=[]
    for fold in folds:
        preds=predict_catalyst(ss, fold, valid_transform, tta=2)
        for ids in tqdm(test_loader):
           all_ids.extend(ids)

        df = pd.DataFrame(
                data=np.concatenate(preds),
                index=all_ids,
                columns=map(str, range(N_CLASSES)))
        df = df.groupby(level=0).mean()  
        print('probs: ', df.head(10)) 
        df = df.reindex(sample_submission.index)
        dfs.append(df)
        print(dfs)          
    df = pd.concat(dfs)
    print(df.head())
    # average 5 folds
    df = df.groupby(level=0).mean() 
    df[:] = binarize_prediction(df.values, num_classes=N_CLASSES, threshold=0.11)
    df = df.apply(get_classes, axis=1)
    df.name = 'attribute_ids'
    df.to_csv('submission2.csv', header=True)
    print(df.head())

    

if __name__ == '__main__':
    torch.backends.cudnn.benchmark = cuda.is_available()
    main()    