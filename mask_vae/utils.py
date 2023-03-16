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
        #BCE = nn.functional.binary_cross_entropy(outputs, targets, reduction="mean")

        return dice_loss

def remove_empty_masks(data_dir):
    "Loads in the masks and save the mask without the empty slices as new_prostaat.mhd"
    
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