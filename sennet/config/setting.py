
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2


#-----------------------------------------------------------------------------------------------
### Config
#-----------------------------------------------------------------------------------------------
class CFG:
    model_name = 'Unet'
    backbone = 'mit_b3'

    DataMode = 'kidney_1'
    
    target_size = 1
    in_channel = 3
    
    image_size = 1024
    input_size = 1024
    train_batch_size = 1
    valid_batch_size = 2
    epochs = 10
    lr = 6e-5
    chopping_percentile = 1e-3
    
    valid_id = 1
    
    train_aug_list = [  
        A.Rotate(limit=270, p=0.5),
        A.RandomScale(scale_limit=(0.8, 1.25), interpolation=cv2.INTER_CUBIC, p=0.05),
        A.RandomCrop(input_size, input_size, p=1),
        A.RandomGamma(p=0.05),
        A.RandomBrightnessContrast(p=0.05),
        A.GaussianBlur(p=0.05),
        A.MotionBlur(p=0.05),
        A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.05),
        ToTensorV2(transpose_mask=True),
    ]
    train_aug = A.Compose(train_aug_list)
    
    valid_aug_list = [
        ToTensorV2(transpose_mask=True),
    ]
    valid_aug = A.Compose(valid_aug_list)