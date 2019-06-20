import configs
import cv2
from PIL import Image
from torchvision.transforms import (
    ToTensor, Normalize, Compose, Resize, CenterCrop, RandomCrop, ColorJitter,
    RandomHorizontalFlip, RandomGrayscale)
import albumentations as A

SIZE = configs.SIZE


train_transform = Compose([
    CenterCrop(SIZE),
    RandomHorizontalFlip(0.5)
    ])


test_transform = Compose([
    CenterCrop(SIZE),
    RandomHorizontalFlip(0.5),
    ])  


tensor_transform = Compose([
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    
albu_transform = A.Compose([
        A.LongestMaxSize(SIZE, interpolation=cv2.INTER_AREA),
        A.PadIfNeeded(SIZE, SIZE, border_mode=cv2.BORDER_CONSTANT),   
        
        A.OneOf([
            A.ShiftScaleRotate(scale_limit=0.2, shift_limit=0, rotate_limit=10, border_mode=cv2.BORDER_CONSTANT),
            A.ElasticTransform(sigma=10, alpha_affine=10, border_mode=cv2.BORDER_CONSTANT),
        ]),

        A.Cutout(),
        A.HorizontalFlip(),
        A.Normalize()
        ])


valid_transform = A.Compose([
        A.LongestMaxSize(SIZE, interpolation=cv2.INTER_AREA),
        A.PadIfNeeded(SIZE, SIZE, border_mode=cv2.BORDER_CONSTANT),       
        A.Normalize()
        ])
 