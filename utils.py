from datetime import datetime
import json
import glob
import os
from pathlib import Path
from typing import Dict
import random
from collections import Counter

import numpy as np
import pandas as pd
from scipy.stats.mstats import gmean
import torch
from torch import nn, cuda



def unzip(path_to_zip_file):
    """   
    helper function to unzip files in the current directory
    """
    import zipfile
    zip_ref = zipfile.ZipFile(path_to_zip_file, 'r')
    zip_ref.extractall() # extract in the current dir, can add desired directory here
    zip_ref.close()
    

def set_seed(seed=1234):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
    

def load_model(model: nn.Module, path: Path) -> Dict:
    state = torch.load(str(path))
    model.load_state_dict(state['model'])
    print('Loaded model from epoch {epoch}, step {step:,}'.format(**state))
    return state


def check_fold(train_fold: pd.DataFrame, valid_fold: pd.DataFrame):
    """
    Check labels distribution in train and validation folds
    : train_fold : dataframe with train meta
    : valid_fold : dataframe with validation meta
    """
    cls_counts = Counter(cls for classes in train_fold['attribute_ids'].str.split()
                         for cls in classes)   
    print('train_fold counts :', cls_counts)
    cls_counts = Counter(cls for classes in valid_fold['attribute_ids'].str.split()
                        for cls in classes) 
    print('valid_fold counts :', cls_counts)


def write_event(log, step: int, **data):
    """Logs output of the model training
    """
    data['step'] = step
    data['dt'] = datetime.now().isoformat()
    log.write(json.dumps(data, sort_keys=True))
    log.write('\n')
    log.flush()

