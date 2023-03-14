import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List, Tuple
from upsample import Upsample
from utils import sample_z

_conv = \
    lambda    in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, : \
    nn.Conv2d(in_channels, out_channels, kernel_size, stride  , padding  , bias=bias)
_batch_norm = nn.InstanceNorm2d
_relu = lambda : nn.LeakyReLU(negative_slope=0.2, inplace=True)
_upsample = lambda : Upsample([1,3,3,1])
_layers = (1,1,1,1)
_chs_e = (32, 64, 128, 256)
_chs_g = (256, 128, 64, 32)
_chs_d = (32, 64, 128, 256)
_z_dim = 256
_spatial_size = (64, 64)

class SPADE(nn.Module):
    def __init__(
            self, 
            nch_norm   : int, 
            nch_segmap : int = 2, 
            nch_hidden : int = 128,
        ):
        super(SPADE, self).__init__()
        self.param_free_norm = _batch_norm(nch_norm)
        self.relu = _relu()
        self.conv_shared = _conv(nch_segmap, nch_hidden, 3, 1, 1)
        self.conv_gamma  = _conv(nch_hidden, nch_norm  , 3, 1, 1)
        self.conv_beta   = _conv(nch_hidden, nch_norm  , 3, 1, 1)

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
    def __init__(
            self, 
            in_ch    : int, 
            out_ch   : int, 
            stride   : int = 1, 
            upsample : int = 1, 
            spade    : bool = False,
        ):
        super(ResidualBlock, self).__init__()
        batch_norm = SPADE if spade else _batch_norm
        self.spade = spade

        self.upsample = (lambda x : x) if upsample==1 else _upsample()
        self.conv1 = _conv(in_ch , out_ch, 3, stride, 1, bias=False)
        self.relu =  _relu()
        self.bn1 = batch_norm(out_ch)
        self.conv2 = _conv(out_ch, out_ch, 3,      1, 1, bias=False)
        self.bn2 = batch_norm(out_ch)

        if stride!=1 or in_ch!=out_ch:
            self.convsc = _conv(in_ch, out_ch, 1, stride, bias=True)
            self.bnsc   = batch_norm(out_ch)
        else:
            self.convsc = (lambda x, *args : x)
            self.bnsc   = (lambda x, *args : x)

    def forward(self, x, segmap=None):
        res = self.upsample(x)
        res = self.convsc(res)
        res = self.bnsc(res, segmap) if self.spade else self.bnsc(res)
        out = self.upsample(x)
        out = self.conv1(out)
        out = self.bn1(out, segmap) if self.spade else self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out, segmap) if self.spade else self.bn2(out)
        out = out + res
        out = self.relu(out)
        return out
    

class Encoder(nn.Module):
    def __init__(
            self, 
            spatial_size : Tuple[int, int] = _spatial_size, 
            z_dim        : int = _z_dim, 
            chs          : Tuple[int, int, int, int] = _chs_e, 
            layers       : Tuple[int, int, int, int] = _layers,
        ):
        super(Encoder, self).__init__()

        self.conv1 = _conv(1, chs[0], 7, 1, 3)
        self.bn1 = _batch_norm(chs[0])
        self.relu1 = _relu()
        
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
      
        self.layer1 = self._make_layer(chs[0], chs[0], layers[0], stride=1)
        self.layer2 = self._make_layer(chs[0], chs[1], layers[1], stride=2)
        self.layer3 = self._make_layer(chs[1], chs[2], layers[2], stride=2)
        self.layer4 = self._make_layer(chs[2], chs[3], layers[3], stride=2)
        self.layers = [self.layer1, self.layer2, self.layer3, self.layer4]

        t = 2**4
        _h, _w = spatial_size[0]//t, spatial_size[1]//t

        self.final = nn.Sequential(nn.Flatten(1), nn.Linear(chs[-1] * _h * _w, 2 * z_dim))

    def _make_layer(self, in_ch : int, out_ch,  blocks : int, stride : int = 2):
        layers = [ResidualBlock(in_ch, out_ch, stride, 1)]
        for i in range(blocks-1):
            layers.append(ResidualBlock(out_ch, out_ch, 1, 1))
        return nn.ModuleList(layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.maxpool(out)
        
        for i, layer in enumerate(self.layers, start=1):
            for block in layer:
                out = block(out)

        out = self.final(out)
        mu, logvar = torch.chunk(out, 2, dim=1)
        return mu, logvar   # 2 chunks, 1 each for mu and logvar
    

class Generator(nn.Module):
    def __init__(
            self, 
            spatial_size : Tuple[int, int] = _spatial_size, 
            z_dim        : int = _z_dim, 
            chs          : Tuple[int, int, int, int] = _chs_g, 
            layers       : Tuple[int, int, int, int] = _layers, 
            spade        : bool = False, 
            tanh         : bool = True,
        ):
        super(Generator, self).__init__()
        self.z_dim = z_dim  
        self.spade = spade
        self.tanh = tanh

        t = 2**4
        _h, _w = spatial_size[0]//t, spatial_size[1]//t
        
        self.proj_z = nn.Linear(
            self.z_dim, chs[0] * _h * _w
        )  
        self.reshape = lambda x : torch.reshape(
            x, (-1, chs[0], _h, _w)
        )  

        self.layer1 = self._make_layer(chs[0], chs[0], layers[0], upsample=2)
        self.layer2 = self._make_layer(chs[0], chs[1], layers[1], upsample=2)
        self.layer3 = self._make_layer(chs[1], chs[2], layers[2], upsample=2)
        self.layer4 = self._make_layer(chs[2], chs[3], layers[3], upsample=2)
        self.layers = [self.layer1, self.layer2, self.layer3, self.layer4]

        self.proj_o = _conv(chs[3], 1, 3, 1, 1)

    def _make_layer(self, in_ch : int, out_ch,  blocks : int, upsample : int = 2):
        layers = [ResidualBlock(in_ch, out_ch, 1, upsample, spade=self.spade)]
        for i in range(blocks-1):
            layers.append(ResidualBlock(out_ch, out_ch, 1, 1, spade=self.spade))
        return nn.ModuleList(layers)

    def forward(self, z, segmap=None):
        out = self.proj_z(z)
        out = self.reshape(out)
        
        for i, layer in enumerate(self.layers, start=1):
            for block in layer:
                out = block(out, segmap)

        out = self.proj_o(out)
        if self.tanh:
            out = torch.tanh(out)
        return out
    

class Discriminator(nn.Module):
    def __init__(
            self, 
            spatial_size : Tuple[int, int] = _spatial_size, 
            chs          : Tuple[int, int, int, int] = _chs_d, 
            layers       : Tuple[int, int, int, int] = _layers,
            l            : int = 2,
        ):
        super(Discriminator, self).__init__()
        self.l = l

        self.conv1 = _conv(1, chs[0], 7, 1, 3)
        self.bn1 = _batch_norm(chs[0])
        self.relu1 = _relu()
        
        
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
      
        self.layer1 = self._make_layer(chs[0], chs[0], layers[0], stride=1)
        self.layer2 = self._make_layer(chs[0], chs[1], layers[1], stride=2)
        self.layer3 = self._make_layer(chs[1], chs[2], layers[2], stride=2)
        self.layer4 = self._make_layer(chs[2], chs[3], layers[3], stride=2)
        self.layers = [self.layer1, self.layer2, self.layer3, self.layer4]

        t = 2**4
        _h, _w = spatial_size[0]//t, spatial_size[1]//t

        self.final = nn.Sequential(nn.Flatten(1), nn.Linear(chs[-1] * _h * _w, 1))

    def _make_layer(self, in_ch : int, out_ch,  blocks : int, stride : int = 2):
        layers = [ResidualBlock(in_ch, out_ch, stride, 1)]
        for i in range(blocks-1):
            layers.append(ResidualBlock(out_ch, out_ch, 1, 1))
        return nn.ModuleList(layers)

    def forward(self, x, features=False):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.maxpool(out)
        
        for i, layer in enumerate(self.layers, start=1):
            for block in layer:
                out = block(out)
            if i == self.l and features:
                return out

        out = self.final(out)
        return out
    

class VAE(nn.Module):
    def __init__(
        self,
        chs_e  : Tuple[int, int, int, int] = _chs_e,
        chs_g  : Tuple[int, int, int, int] = _chs_g,
        layers : Tuple[int, int, int, int] = _layers,
        spade  : bool = False,
        tanh   : bool = True,
    ):
        super(VAE, self).__init__()
        self.encoder = Encoder(chs=chs_e, layers=layers)
        self.generator = Generator(chs=chs_g, layers=layers, spade=spade, tanh=tanh)

    def forward(self, x, segmap=None):
        mu, logvar = self.encoder(x)
        latent_z = sample_z(mu, logvar)
        recons = self.generator(latent_z, segmap)
        
        return recons, mu, logvar
    

class VAEGAN(nn.Module):
    def __init__(
        self,
        chs_e  : Tuple[int, int, int, int] = _chs_e,
        chs_g  : Tuple[int, int, int, int] = _chs_g,
        chs_d  : Tuple[int, int, int, int] = _chs_d,
        layers : Tuple[int, int, int, int] = _layers,
        spade  : bool = False,
        tanh   : bool = True,
    ):
        super(VAEGAN, self).__init__()
        self.encoder = Encoder(chs=chs_e, layers=layers)
        self.generator = Generator(chs=chs_g, layers=layers, spade=spade, tanh=tanh)
        self.discriminator = Discriminator(chs=chs_d, layers=layers)

    def forward(self, x, segmap=None):
        mu, logvar = self.encoder(x)
        latent_z = sample_z(mu, logvar)
        recons = self.generator(latent_z, segmap)
        
        scores_real = self.discriminator(x)
        features_real = self.discriminator(x, True)
        
        scores_fake = self.discriminator(recons)
        features_fake = self.discriminator(recons, True)

        return recons, mu, logvar, scores_real, features_real, scores_fake, features_fake
