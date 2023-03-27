import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

import numpy as np
import SimpleITK as sitk
from tqdm.auto import tqdm, trange
import os 

class Prostate2D(Dataset):
    def __init__(self, indx, data_dir):
        self.images = []
        self.labels = []
        criterium = lambda path : path[:4] + path[6:] == "Case.nii.gz" and int(path[4:6]) in indx
        listdir = os.listdir(data_dir)
        for path in tqdm(list(filter(criterium, listdir))):
            path_img = os.path.join(data_dir, path)
            img = sitk.GetArrayFromImage(sitk.ReadImage(path_img)).astype(np.float32)
            img = torch.from_numpy(img)[:,None]
            p99 = torch.quantile(img, 0.99)
            p01 = torch.quantile(img, 0.01)
            
            img[img>p99] = p99
            img[img<p01] = p01
            img = (img - p01)/(p99 - p01)*2.0 - 1.0

            img = F.interpolate(img, size=(64,64), mode='bilinear', antialias=True)

            path_mask = os.path.join(data_dir, path[:6] + "_segmentation" + path[6:])
            mask = sitk.GetArrayFromImage(sitk.ReadImage(path_mask)).astype(np.float32)
            mask = torch.from_numpy(mask)[:,None]
            mask = mask.repeat_interleave(2, dim=1)

            mask = F.interpolate(mask, size=(64,64), mode='nearest')
            mask[:,1] = 1 - mask[:,1]

            assert img.shape[0] == mask.shape[0]
            self.images.append(img)
            self.labels.append(mask)
        self.images = torch.cat(self.images, dim=0)
        self.labels = torch.cat(self.labels, dim=0)

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        images = self.images[idx]
        labels = self.labels[idx]
        return images, labels