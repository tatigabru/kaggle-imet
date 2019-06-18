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


class TrainDataset(Dataset):
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
                
        if height < 320:
            ratio = 400/height
            image = image.resize((int(width * ratio), int(height * ratio)), Image.ANTIALIAS)            
        if width < 320:
            ratio = 400/width
            image = image.resize((int(width * ratio), int(height * ratio)), Image.ANTIALIAS)     
            
        image = self.image_transform(image)
        image = tensor_transform(image)
        # labels encoding
        target = torch.zeros(N_CLASSES)
        for cls in item.attribute_ids.split():
            target[int(cls)] = 1
        return image, target


class TestDataset(Dataset):
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
        
        if height < 320:
            ratio = 400/height
            image = image.resize((int(width * ratio), int(height * ratio)), Image.ANTIALIAS)            
        if width < 320:
            ratio = 400/width
            image = image.resize((int(width * ratio), int(height * ratio)), Image.ANTIALIAS)     
            
        image = self.image_transform(image)
        image = tensor_transform(image)    
        return image, item.id
    

class TTADataset(Dataset):
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
       
        if height < 320:
            ratio = 330/height
            image = image.resize((int(width * ratio), int(height * ratio)), Image.ANTIALIAS)            
        if width < 320:
            ratio = 330/width
            image = image.resize((int(width * ratio), int(height * ratio)), Image.ANTIALIAS)     
            
        image = self.image_transform(image)
        image = tensor_transform(image)             
        return image, item.id


def load_image(item, root: Path) -> Image.Image:
    image = cv2.imread(str(root / f'{item.id}.png'))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(image)


def get_ids(root: Path) -> List[str]:
    return sorted({p.name.split('_')[0] for p in root.glob('*.png')})
