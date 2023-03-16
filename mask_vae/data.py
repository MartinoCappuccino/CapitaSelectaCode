import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.transforms as transforms


import numpy as np
import SimpleITK as sitk
from tqdm.auto import tqdm, trange
import os 

class ProstateMRMaskDataset(torch.utils.data.Dataset):
    """Dataset containing prostate MR images.

    Parameters
    ----------
    paths : list[Path]
        paths to the patient data
    img_size : list[int]
        size of images to be interpolated to
    empty_masks: boleaan
        option include or exclude empty masks
        False for no empty mask
    """

    def __init__(self, paths, img_size, empty_masks = False):
        self.labels = []
        # load images
        
        if empty_masks == False: 
            prostaat = "new_prostaat.mhd"
        else: 
            prostaat = "prostaat.mhd"
            
        for path in paths:

            mask = sitk.GetArrayFromImage(sitk.ReadImage(path / prostaat)).astype(np.float32)
            mask = torch.from_numpy(mask)[:,None]
            mask = mask.repeat_interleave(2, dim=1)

            mask = F.interpolate(mask, size=img_size, mode='nearest')
            mask[:,1] = 1 - mask[:,1]

            self.labels.append(mask)
        self.labels = torch.cat(self.labels, dim=0)

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, index):
        labels = self.labels[index]
        return labels
    
class ProstateMRDataset(torch.utils.data.Dataset):
    """Dataset containing prostate MR images.

    Parameters
    ----------
    paths : list[Path]
        paths to the patient data
    img_size : list[int]
        size of images to be interpolated to
    """

    def __init__(self, paths, img_size):
        self.images = []
        self.labels = []
        # load images
        
        for path in paths:
            img = sitk.GetArrayFromImage(sitk.ReadImage(path / "mr_bffe.mhd")).astype(np.float32)
            img = torch.from_numpy(img)[:,None]
            p99 = torch.quantile(img, 0.99)
            p01 = torch.quantile(img, 0.01)
            
            img[img>p99] = p99
            img[img<p01] = p01
            img = (img - p01)/(p99 - p01)*2.0 - 1.0

            img = F.interpolate(img, size=img_size, mode='bilinear', antialias=True)

            mask = sitk.GetArrayFromImage(sitk.ReadImage(path / "prostaat.mhd")).astype(np.float32)
            mask = torch.from_numpy(mask)[:,None]
            mask = mask.repeat_interleave(2, dim=1)

            mask = F.interpolate(mask, size=img_size, mode='nearest')
            mask[:,1] = 1 - mask[:,1]

            #assert img.shape[0] == mask.shape[0]
            self.images.append(img)
            self.labels.append(mask)
        self.images = torch.cat(self.images, dim=0)
        self.labels = torch.cat(self.labels, dim=0)

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, index):
        images = self.images[index]
        labels = self.labels[index]
        return images, labels
    
