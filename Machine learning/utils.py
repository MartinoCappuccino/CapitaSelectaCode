import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import SimpleITK as sitk
import os 
import numpy as np 
from tqdm.auto import tqdm, trange

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

def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)

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
    
class ProstateMRUNETDataset(torch.utils.data.Dataset):
    """Dataset containing prostate MR images.

    Parameters
    ----------
    paths : list[Path]
        paths to the patient data
    img_size : list[int]
        size of images to be interpolated to
    """

    def __init__(self, paths, img_size, mask_generator, image_generator, ratio, validation=True):
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

        if validation == False:
            for i in range(int(len(self.images)*ratio)):
                new_mask = mask_generator(self.labels[i])
                new_image = image_generator(new_mask)
                self.images = torch.cat(new_image, dim=0)
                self.labels = torch.cat(new_mask, dim=0)

    def __len__(self):
        """Returns length of dataset"""
        return self.images.shape[0]

    def __getitem__(self, index):
        images = self.images[index]
        labels = self.labels[index]
        return images, labels

class DiceBCELoss(nn.Module):
    """Loss function, computed as the sum of Dice score and binary cross-entropy.

    Notes
    -----
    This loss assumes that the inputs are logits (i.e., the outputs of a linear layer),
    and that the targets are integer values that represent the correct class labels.
    """

    def __init__(self):
        super(DiceBCELoss, self).__init__()

    def forward(self, outputs, targets, smooth=1):
        """Calculates segmentation loss for training

        Parameters
        ----------
        outputs : torch.Tensor
            predictions of segmentation model
        targets : torch.Tensor
            ground-truth labels
        smooth : float
            smooth parameter for dice score avoids division by zero, by default 1

        Returns
        -------
        float
            the sum of the dice loss and binary cross-entropy
        """
        outputs = torch.sigmoid(outputs)

        # flatten label and prediction tensors
        outputs = outputs.view(-1)
        targets = targets.view(-1)

        # compute Dice
        intersection = (outputs * targets).sum()
        dice_loss = 1 - (2.0 * intersection + smooth) / (
            outputs.sum() + targets.sum() + smooth
        )
        BCE = nn.functional.binary_cross_entropy(outputs, targets, reduction="mean")

        return BCE + dice_loss
