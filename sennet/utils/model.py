
from config.setting import CFG
from dotenv import load_dotenv

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DataParallel 
from torch.cuda.amp import autocast

import segmentation_models_pytorch as smp

import albumentations as A
from albumentations.pytorch import ToTensorV2


#-----------------------------------------------------------------------------------------------
### Model
#-----------------------------------------------------------------------------------------------
class CustomModel(nn.Module):
    def __init__(self, CFG, weight=None):
        super().__init__()
        self.model = smp.Unet(
            encoder_name = CFG.backbone,
            encoder_weights = weight,
            in_channels = CFG.in_channel,
            classes = CFG.target_size,
            activation = None,
        )
        
    def forward(self, image):
        output = self.model(image)
        return output[:,0]
    
    
def build_model(weight="imagenet"):
    load_dotenv()
    
    print('model_name', CFG.model_name)
    print('backbone', CFG.backbone)
    
    model = CustomModel(CFG, weight)
    
    return model.cuda()