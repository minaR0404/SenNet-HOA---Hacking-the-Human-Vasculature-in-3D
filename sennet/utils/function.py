
import numpy as np
import torch


#-----------------------------------------------------------------------------------------------
### utils
#-----------------------------------------------------------------------------------------------
def min_max_normalization(x):
    shape = x.shape
    
    if x.ndim > 2:
        x = x.reshape(x.shape[0], -1)
        
    min_ = x.min(dim=-1, keepdim=True)[0]
    max_ = x.max(dim=-1, keepdim=True)[0]
    
    if min_.mean() == 0 and max_.mean() == 1:
        return x.reshape(shape)
    
    x = (x - min_) / (max_ - min_ + 1e-9)
    
    return x.reshape(shape)


def norm_with_clip(x, smooth=1e-5):
    dim = list(range(1, x.ndim))
    mean = x.mean(dim=dim, keepdim=True)
    std = x.std(dim=dim, keepdim=True)
    
    x = (x - mean) / (std + smooth)
    x[x>5] = (x[x>5] - 5) * 1e-3 + 5
    x[x<-3] = (x[x<-3] + 3) * 1e-3 - 3
    
    return x


def add_noise(x, max_randn_rate=0.1, randn_rate=None, x_already_normed=False):
    ndim = x.ndim - 1
    if x_already_normed:
        x_std = torch.ones([x.shape[0]] + [1] * ndim, device=x.device, dtype=x.dtype)
        x_mean = torch.zeros([x.shape[0]] + [1] * ndim, device=x.device, dtype=x.dtype)
    else:
        dim = list(range(1, x.ndim))
        x_std = x.std(dim=dim, keepdim=True)
        x_mean = x.mean(dim=dim, keepdim=True)
        
    if randn_rate is None:
        randn_rate = max_randn_rate * np.random.rand() * torch.rand(x_mean.shape, device=x.device, dtype=x.dtype)
        
    cache = (x_std**2 + (x_std * randn_rate)**2)**0.5
    x = (x - x_mean + torch.rand(size=x.shape, device=x.device, dtype=x.dtype) * randn_rate * x_std) / (cache + 1e-7)
    
    return x