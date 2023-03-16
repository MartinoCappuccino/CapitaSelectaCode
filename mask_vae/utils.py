import torch
import torch.nn as nn
import torch.nn.functional as F
import SimpleITK as sitk
import os 
import numpy as np 


def get_noise(n_samples, z_dim, device="cpu"):
    return torch.randn(n_samples, z_dim, device=device)

def sample_z(mu, logvar):
    eps = torch.randn(mu.size(), device=mu.device).to(mu.dtype)
    return (logvar / 2).exp() * eps + mu

def kld_loss(mu, logvar):
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

def dice_loss(inputs, recons, eps=1e-5):
    intersection = torch.sum(inputs * recons)
    divider = torch.sum(inputs) + torch.sum(recons)
    dice = 2.0 * intersection + eps
    dice /= divider + eps
    return 1 - dice

def remove_empty_masks(data_dir):
    
    patients = [
    path 
    for path in data_dir.glob("*")
    if not any(part.startswith(".") for part in path.parts)
    ]
    
    for patient in patients:
        img = sitk.GetArrayFromImage(sitk.ReadImage(patient / "prostaat.mhd"))
        new_img = []
        for i in range(86):
            if  (img[i,:,:] == 1).any():
                new_img.append(img[i,:,:]) 
            
        img_array = np.array(new_img)
        new_prostaat = sitk.GetImageFromArray(img_array)
        sitk.WriteImage(new_prostaat, os.path.join(data_dir, patient, "new_prostaat.mhd"))