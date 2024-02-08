
import numpy as np
from config.setting import CFG


#-----------------------------------------------------------------------------------------------
### Size(1024)
#-----------------------------------------------------------------------------------------------
def to_1024(img, image_size=1024):
    if image_size > img.shape[1]:
        img = np.rot90(img)
        start1 = (CFG.image_size - img.shape[0]) // 2
        top = img[0:start1, 0:img.shape[1]]
        bottom = img[img.shape[0]-start1:img.shape[0], 0:img.shape[1]]
        img_result = np.concatenate((top, img, bottom), axis=0)
        img_result = np.rot90(img_result)
        img_result = np.rot90(img_result)
        img_result = np.rot90(img_result)
    else:
        img_result = img
    return img_result


def to_1024_no_rot(img, image_size=1024):
    if image_size > img.shape[0]:
        start1 = (image_size - img.shape[0]) // 2
        top = img[0:start1, 0:img.shape[1]]
        bottom = img[img.shape[0]-start1:img.shape[0], 0:img.shape[1]]
        img_result = np.concatenate((top, img, bottom), axis=0)
    else:
        img_result = img
    return img_result


def to_1024_1024(img, image_size=1024):
    img_result = to_1024(img, image_size)
    return img_result


def to_original(im_after, img, image_size=1024):
    top_ = 0
    left_ = 0
    
    if (im_after.shape[0] > img.shape[0]):
        top_ = (image_size - img.shape[0]) // 2
    if (im_after.shape[1] > img.shape[1]):
        left_ = (image_size - img.shape[1]) // 2
        
    if (top_ > 0) or (left_ > 0):
        img_result = im_after[top_:img.shape[0]+top_, left_:img.shape[1]+left_]
    else:
        img_result = im_after
    return img_result