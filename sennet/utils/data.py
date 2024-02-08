
import numpy as np
import cv2
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

from config.setting import CFG
from .function import min_max_normalization
from .size import to_1024_1024

#-----------------------------------------------------------------------------------------------
### Data
#-----------------------------------------------------------------------------------------------
class Data_loader(Dataset):
    def __init__(self, paths, is_label):
        self.paths = paths
        self.paths.sort()
        self.is_label = is_label
        
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, index):
        img = cv2.imread(self.paths[index], cv2.IMREAD_GRAYSCALE)
        if CFG.input_size == 1024 and CFG.image_size == 1024:
            img = to_1024_1024(img, image_size=CFG.image_size)  # to_original(im_after, img_save, image_size=1024)
            img = torch.from_numpy(img.copy())
        else: 
            img = torch.from_numpy(img)
        
        if self.is_label:
            img = (img!=0).to(torch.uint8) * 255
        else:
            img = img.to(torch.uint8)
            
        return img
    
    
def load_data(paths, is_label=False):
    data_loader = Data_loader(paths, is_label)
    data_loader = DataLoader(data_loader, 
                             batch_size = 16, 
                             num_workers = 2)
    data = []
    for x in tqdm(data_loader):
        data.append(x)
        
    x = torch.cat(data, dim=0)
    del data
    
    if not is_label:
        TH = x.reshape(-1).numpy()
        index = -int(len(TH) * CFG.chopping_percentile)
        TH = np.partition(TH, index)[index]
        x[x>TH] = int(TH)
        
        TH = x.reshape(-1).numpy()
        index = -int(len(TH) * CFG.chopping_percentile)
        TH = np.partition(TH, -index)[-index]
        x[x<TH] = int(TH)
        
        x = min_max_normalization(x.to(torch.float16)[None])[0] * 255
        x = x.to(torch.uint8)
        
    return x


class Kaggle_Dataset(Dataset):
    def __init__(self, x, y, arg=False):
        super(Dataset, self).__init__()
        
        self.x = x
        self.y = y
        self.image_size= CFG.image_size
        self.in_channels = CFG.in_channel
        self.arg = arg
        
        if arg:
            self.transform = CFG.train_aug
        else:
            self.transform = CFG.valid_aug
            
    def __len__(self):
        return sum([y.shape[0] - self.in_channels for y in self.y])
    
    def __getitem__(self, index):
        idx = 0
        for x in self.x:
            if index > x.shape[0] - self.in_channels:
                index -= x.shape[0] - self.in_channels
                idx += 1
            else:
                break
        
        x = self.x[idx]
        y = self.y[idx]
        
        if CFG.image_size == 1024 and CFG.input_size == 1024:
            x_index = (x.shape[1] - self.image_size) // 2 
            y_index = (x.shape[2] - self.image_size) // 2
        else:
            x_index = np.random.randint(0, x.shape[1] - self.image_size)
            y_index = np.random.randint(0, x.shape[2] - self.image_size)
        
        x = x[index:index+self.in_channels, x_index:x_index+self.image_size, y_index:y_index+self.image_size]
        y = y[index+self.in_channels//2, x_index:x_index+self.image_size, y_index:y_index+self.image_size]
        
        data = self.transform(image=x.numpy().transpose(1,2,0), mask=y.numpy())
        
        x = data["image"]
        y = data["mask"] >= 127
        
        if self.arg:
            dr = np.random.randint(4)
            x = x.rot90(dr, dims=(1,2))
            y = y.rot90(dr, dims=(0,1))
            for dm in range(3):
                if np.random.randint(2):
                    x = x.flip(dims=(dm, ))
                    if dm >= 1:
                        y = y.flip(dims=(dm-1, ))
                        
        return x, y