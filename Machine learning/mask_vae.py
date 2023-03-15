# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 16:21:59 2023

@author: 20182371
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Callable, List, Tuple, Type, Union
from torchvision import transforms
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm, trange
import os
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.model_selection import train_test_split
from pathlib import Path
from torchvision.utils import make_grid, save_image
import copy
from torchvision.datasets import FashionMNIST

from torchmetrics import StructuralSimilarityIndexMeasure

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

KLD_WEIGHT = 1e-4
SSIM_WEIGHT = 0.5

def upfirdn2d_native(input, kernel, up, down, pad):
    up_x, up_y = up[0], up[1]
    down_x, down_y = down[0], down[1]
    pad_x0, pad_x1, pad_y0, pad_y1 = pad[0], pad[1], pad[2], pad[3]
    _, channel, in_h, in_w = input.shape
    input = input.reshape(-1, in_h, in_w, 1)

    _, in_h, in_w, minor = input.shape
    kernel_h, kernel_w = kernel.shape

    out = input.view(-1, in_h, 1, in_w, 1, minor)
    out = F.pad(out, [0, 0, 0, up_x - 1, 0, 0, 0, up_y - 1], mode = 'reflect')
    out = out.view(-1, in_h * up_y, in_w * up_x, minor)

    out = F.pad(
        out, 
        [0, 0, max(pad_x0, 0), max(pad_x1, 0), max(pad_y0, 0), max(pad_y1, 0)], 
        mode = 'reflect',
    )
    out = out[
        :,
        max(-pad_y0, 0) : out.shape[1] - max(-pad_y1, 0),
        max(-pad_x0, 0) : out.shape[2] - max(-pad_x1, 0),
        :,
    ]

    out = out.permute(0, 3, 1, 2)
    out = out.reshape(
        [-1, 1, in_h * up_y + pad_y0 + pad_y1, in_w * up_x + pad_x0 + pad_x1]
    )
    w = torch.flip(kernel, [0, 1]).view(1, 1, kernel_h, kernel_w)
    out = F.conv2d(out, w)
    out = out.reshape(
        -1,
        minor,
        in_h * up_y + pad_y0 + pad_y1 - kernel_h + 1,
        in_w * up_x + pad_x0 + pad_x1 - kernel_w + 1,
    )
    out = out.permute(0, 2, 3, 1)
    out = out[:, ::down_y, ::down_x, :]

    out_h = (in_h * up_y + pad_y0 + pad_y1 - kernel_h) // down_y + 1
    out_w = (in_w * up_x + pad_x0 + pad_x1 - kernel_w) // down_x + 1

    return out.view(-1, channel, out_h, out_w)

def upfirdn2d(input, kernel, up=1, down=1, pad=(0, 0)):
    out = upfirdn2d_native(
        input, kernel, (up, up), (down, down), (pad[0], pad[1], pad[0], pad[1])
    )

    return out

def make_kernel(k):
    k = torch.tensor(k, dtype=torch.float32)

    if k.ndim == 1:
        k = k[None, :] * k[:, None]

    k /= k.sum()

    return k

class Upsample(nn.Module):
    def __init__(self, kernel, factor=2):
        super().__init__()

        self.factor = factor
        kernel = make_kernel(kernel) * (factor ** 2)
        self.register_buffer('kernel', kernel)

        p = kernel.shape[0] - factor

        pad0 = (p + 1) // 2 + factor - 1
        pad1 = p // 2

        self.pad = (pad0, pad1)

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, up=self.factor, down=1, pad=self.pad)

        return out

CONV = \
lambda in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, : \
    nn.Conv2d(in_channels, out_channels, kernel_size, stride  , padding  , bias=bias, padding_mode="reflect")
BATCH_NORM = nn.InstanceNorm2d
RELU = lambda : nn.LeakyReLU(negative_slope=0.2, inplace=True)
UPSAMPLE = lambda : Upsample([1,3,3,1])
#DOWNSAMPLE = lambda : Downsample([1,3,3,1])
LAYERS = (1,1,1,1)
CHS_E = (32, 64, 128, 256)
CHS_D = (256, 128, 64, 32)
Z_DIM = 256

SSIM = StructuralSimilarityIndexMeasure(data_range=1, kernel_size=7).to(device)
LOSS = nn.L1Loss()

KLD_EPOCHS = 75

class SPADE(nn.Module):
    def __init__(self, nch_norm : int, nch_segmap : int = 2, nch_hidden : int = 128):
        super(SPADE, self).__init__()
        self.param_free_norm = BATCH_NORM(nch_norm)
        self.relu = RELU()
        self.conv_shared = CONV(nch_segmap, nch_hidden, 3, 1, 1)
        self.conv_gamma  = CONV(nch_hidden, nch_norm  , 3, 1, 1)
        self.conv_beta   = CONV(nch_hidden, nch_norm  , 3, 1, 1)

    def forward(self, x, segmap):
        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x)

        # Part 2. produce scaling and bias conditioned on semantic map
        segmap = F.interpolate(segmap, size=x.size()[2:], mode='bilinear', antialias=True)
        actv  = self.relu(self.conv_shared(segmap))
        gamma = self.conv_gamma(actv)
        beta  = self.conv_beta(actv)

        # apply scale and bias
        out = normalized * (1 + gamma) + beta

        return out

class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, upsample=1, spade=False):
        super(ResidualBlock, self).__init__()
        batch_norm = SPADE if spade else BATCH_NORM
        self.spade = spade

        self.upsample = (lambda x : x) if upsample==1 else UPSAMPLE()
        self.conv1 = CONV(in_ch , out_ch, 3, stride, 1, bias=False)
        self.relu =  RELU()
        self.bn1 = batch_norm(out_ch)
        self.conv2 = CONV(out_ch, out_ch, 3,      1, 1, bias=False)
        self.bn2 = batch_norm(out_ch)

        self.shortcut = []
        if stride!=1 or in_ch!=out_ch:
            self.shortcut.append(CONV(in_ch, out_ch, 1, stride, bias=True))
            self.shortcut.append(BATCH_NORM(out_ch))
        self.shortcut = nn.Sequential(*self.shortcut)

    def forward(self, x, segmap=None):
        residual = self.upsample(x)
        residual = self.shortcut(residual)
        out = self.upsample(x)
        out = self.conv1(out)
        out = self.bn1(out, segmap) if self.spade else self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out, segmap) if self.spade else self.bn2(out)
        out = out + residual
        out = self.relu(out)
        return out

class Encoder(nn.Module):
    def __init__(self, spatial_size=[64, 64], z_dim=Z_DIM, chs=CHS_E, layers=LAYERS):
        super().__init__()

        self.conv1 = CONV(1, chs[0], 7, 1, 3)
        self.bn1 = BATCH_NORM(chs[0])
        self.relu1 = RELU()
        
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
      
        self.layers = []
        self.layers.extend(self._make_layer(chs[0], chs[0], layers[0], stride=1))
        self.layers.extend(self._make_layer(chs[0], chs[1], layers[1], stride=2))
        self.layers.extend(self._make_layer(chs[1], chs[2], layers[2], stride=2))
        self.layers.extend(self._make_layer(chs[2], chs[3], layers[3], stride=2))
        self.layers = nn.ModuleList(self.layers)

        t = 2**4
        _h, _w = spatial_size[0]//t, spatial_size[1]//t

        self.final = nn.Sequential(nn.Flatten(1), nn.Linear(chs[-1] * _h * _w, 2 * z_dim))

    def _make_layer(self, in_ch : int, out_ch,  blocks : int, stride : int = 2):
        layers = [ResidualBlock(in_ch, out_ch, stride, 1)]
        for i in range(blocks-1):
            layers.append(ResidualBlock(out_ch, out_ch, 1, 1))
        return layers

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.maxpool(out)
        
        for layer in self.layers:
            out = layer(out)

        out = self.final(out)
        return torch.chunk(out, 2, dim=1)  # 2 chunks, 1 each for mu and logvar


class Generator(nn.Module):
    def __init__(self, z_dim=Z_DIM, chs=CHS_D, layers=LAYERS, h=4, w=4, tanh=True, spade=False):

        super().__init__()
        self.spade = spade
        self.tanh = tanh
        self.z_dim = z_dim  

        self.proj_z = nn.Linear(
            self.z_dim, chs[0] * h * w
        )  
        self.reshape = lambda x : torch.reshape(
            x, (-1, chs[0], h, w)
        )  

        self.layers = []
        self.layers.extend(self._make_layer(chs[0], chs[0], layers[0], upsample=2))
        self.layers.extend(self._make_layer(chs[0], chs[1], layers[1], upsample=2))
        self.layers.extend(self._make_layer(chs[1], chs[2], layers[2], upsample=2))
        self.layers.extend(self._make_layer(chs[2], chs[3], layers[3], upsample=2))
        self.layers = nn.ModuleList(self.layers)

        self.proj_o = CONV(chs[3], 1, 3, 1, 1)

    def _make_layer(self, in_ch : int, out_ch,  blocks : int, upsample : int = 2):
        layers = [ResidualBlock(in_ch, out_ch, 1, upsample, spade=self.spade)]
        for i in range(blocks-1):
            layers.append(ResidualBlock(out_ch, out_ch, 1, 1, spade=self.spade))
        return nn.Sequential(*layers)

    def forward(self, z, segmap=None):
        out = self.proj_z(z)
        out = self.reshape(out)
        
        for layer in self.layers:
            out = layer(out, segmap)

        out = self.proj_o(out)
        if self.tanh:
            out = torch.tanh(out)
        return out


class VAE(nn.Module):
    """A representation of the VAE
    Parameters
    ----------
    enc_chs : tuple 
        holds the number of input channels of each block in the encoder
    dec_chs : tuple 
        holds the number of input channels of each block in the decoder
    """
    def __init__(
        self,
        enc_chs=CHS_E,
        dec_chs=CHS_D,
        tanh = True,
        spade = False,
    ):
        super().__init__()
        self.encoder = Encoder(chs=enc_chs)
        self.generator = Generator(chs=dec_chs, tanh=tanh, spade=spade)

    def forward(self, x, segmap=None):
        """Performs a forwards pass of the VAE and returns the reconstruction
        and mean + logvar.
        Parameters
        ----------
        x : torch.Tensor
            the input to the encoder
        Returns
        -------
        torch.Tensor
            the reconstruction of the input image
        float
            the mean of the latent distribution
        float
            the log of the variance of the latent distribution
        """
        mu, logvar = self.encoder(x)
        latent_z = sample_z(mu, logvar)

        return self.generator(latent_z, segmap), mu, logvar


def get_noise(n_samples, z_dim, device="cpu"):
    """Creates noise vectors.
    
    Given the dimensions (n_samples, z_dim), creates a tensor of that shape filled with 
    random numbers from the normal distribution.
    Parameters
    ----------
    n_samples : int
        the number of samples to generate
    z_dim : int
        the dimension of the noise vector
    device : str
        the type of the device, by default "cpu"
    """
    return torch.randn(n_samples, z_dim, device=device)


def sample_z(mu, logvar):
    """Samples noise vector from a Gaussian distribution with reparameterization trick.
    Parameters
    ----------
    mu : float
        the mean of the distribution
    logvar : float
        the log of the variance of the distribution
    """
    eps = torch.randn(mu.size(), device=mu.device).to(mu.dtype)
    return (logvar / 2).exp() * eps + mu


def kld_loss(mu, logvar):
    """Computes the KLD loss given parameters of the predicted 
    latent distribution.
    Parameters
    ----------
    mu : float
        the mean of the distribution
    logvar : float
        the log of the variance of the distribution
    Returns
    -------
    float
        the kld loss
    """
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

def dice_loss(inputs, recons, eps=1e-5):
    intersection = torch.sum(inputs * recons)
    divider = torch.sum(inputs) + torch.sum(recons)
    dice = 2.0 * intersection + eps
    dice /= divider + eps
    return 1 - dice

def vae_loss(
    inputs, 
    recons, 
    mu, 
    logvar, 
    kld_weight=KLD_WEIGHT, 
    ssim_weight=SSIM_WEIGHT, 
    binary=False
):
    """Computes the VAE loss, sum of reconstruction and KLD loss
    Parameters
    ----------
    inputs : torch.Tensor
        the input images to the vae
    recons : torch.Tensor
        the predicted reconstructions from the vae
    mu : float
        the predicted mean of the latent distribution
    logvar : float
        the predicted log of the variance of the latent distribution
    Returns
    -------
    float
        sum of reconstruction and KLD loss
    """
    if binary:
        rec_error = dice_loss(inputs, recons) 
    else:
        rec_error = LOSS(inputs, recons)
        rec_error += (1. - SSIM(inputs/2.0+0.5, recons/2.0+0.5)) * ssim_weight
    kld = kld_loss(mu, logvar)
    return rec_error + kld * kld_weight

