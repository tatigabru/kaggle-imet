import configs
from pathlib import Path
from typing import Callable, List
import cv2
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from transforms import tensor_transform

ON_KAGGLE = configs.ON_KAGGLE
N_CLASSES = configs.NUM_CLASSES
DATA_ROOT = configs.DATA_ROOT
SIZE      = configs.SIZE


class ImetDataset(Dataset):
    def __init__(self, root: Path, df: pd.DataFrame,
                 image_transform: Callable):
        super().__init__()
        self.root = root
        self.df = df
        self.image_transform = image_transform
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        item = self.df.iloc[idx]
        image = load_image(item, self.root)        
        width, height = image.size          
                      
        if self.image_transform is not None:     
            image = self.image_transform(image)  
        image.resize((SIZE, SIZE), Image.ANTIALIAS)             
        image = tensor_transform(image)        
        # labels encoding
        target = torch.zeros(N_CLASSES)
        for cls in item.attribute_ids.split():
            target[int(cls)] = int(1)               
        return image, target
  

class ImetDatasetTTA(Dataset):
    def __init__(self, root: Path, df: pd.DataFrame,
                 image_transform: Callable, tta: int):
        super().__init__()
        self.root = root
        self.df = df
        self.image_transform = image_transform
        self.tta = tta

    def __len__(self):
        return len(self.df) * self.tta

    def __getitem__(self, idx):
        item = self.df.iloc[idx % len(self.df)]
        image = load_image(item, self.root)        
        width, height = image.size        
                
        if self.image_transform is not None:     
            image = self.image_transform(image)              
        image.resize((SIZE, SIZE), Image.ANTIALIAS)
        image = tensor_transform(image)             
        return image, item.id


def load_image(item, root: Path) -> Image.Image:
    image = cv2.imread(str(root / f'{item.id}.png'))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(image)