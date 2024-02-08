
import torch
import torch.nn as nn

import segmentation_models_pytorch as smp


#-----------------------------------------------------------------------------------------------
### Loss
#-----------------------------------------------------------------------------------------------
JaccardLoss = smp.losses.JaccardLoss(mode='binary')
BCELoss     = smp.losses.SoftBCEWithLogitsLoss()
LovaszLoss  = smp.losses.LovaszLoss(mode='binary')
TverskyLoss = smp.losses.TverskyLoss(mode='binary')


def loss_func(y_pred, y_true):
    return 0.75 * LovaszLoss(y_pred, y_true) + 0.25 * BCELoss(y_pred, y_true)


def dice_coef(y_pred, y_true, thr=0.5, dim=(-1,-2), epsilon=0.001):
    y_pred = y_pred.sigmoid()
    y_pred = (y_pred > thr).to(torch.float32)
    y_true = y_true.to(torch.float32)
    
    inter = (y_true * y_pred).sum(dim=dim)
    den = y_true.sum(dim=dim) + y_pred.sum(dim=dim)
    
    dice = ((2 * inter + epsilon) / (den + epsilon)).mean()
    
    return dice


class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()
        
    def forward(self, inputs, targets, smooth=1):
        inputs = inputs.sigmoid()  # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        
        return 1 - dice