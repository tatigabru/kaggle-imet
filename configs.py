# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 00:30:28 2019

configs.py
 
"""
from pathlib import Path

ON_KAGGLE = False # boolean flag, if train on kaggle kernels

DATA_ROOT = Path('../input/imet-2019-fgvc6' if ON_KAGGLE else '../data') # root to data

NUM_CLASSES = 1103 # number of classes

SIZE = 224 # image size for training

