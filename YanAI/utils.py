import torch
import torch.nn as nn
import torch.nn.functional as F

def get_noise(n_samples, z_dim, device="cpu"):
    return torch.randn(n_samples, z_dim, device=device)

def sample_z(mu, logvar):
    eps = torch.randn(mu.size(), device=mu.device).to(mu.dtype)
    return (logvar / 2).exp() * eps + mu

def kld_loss(mu, logvar):
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)
