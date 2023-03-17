import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import make_grid

import numpy as np
import matplotlib.pyplot as plt

from networks import VAE
from utils import kld_loss, get_noise
from typing import Tuple, Callable, List, Union
from pathlib import Path
from utils import dice_loss
from torch.utils.tensorboard import SummaryWriter

from tqdm.auto import tqdm, trange

class TrainerMaskBase():
    loss_names : Tuple[str] = ()
    def __init__(
        self,
        net,
        train_loader,
        valid_loader,
        device = "cpu",
    ):
        self.net = net
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.device = device

        self.optimizers : List[torch.optim.Optimizer] 
        self.train_losses = [[] for i in range(len(self.loss_names))]
        self.valid_losses = [[] for i in range(len(self.loss_names))]

    def train_step(self, masks : torch.Tensor) -> Tuple[float]:
        raise NotImplementedError
    
    def valid_step(self, masks : torch.Tensor) -> Tuple[float]:
        raise NotImplementedError

    def train_epoch(self) -> Tuple[float]:
        self.net.train()
        epoch_losses = np.zeros((len(self.loss_names),), dtype=np.float32)
        n = 0
        for masks in tqdm(self.train_loader):
            masks  =  masks.to(self.device)
            losses = self.train_step(masks)
            self.nstep += 1
            n += masks.shape[0]
            epoch_losses += np.array(losses) * masks.shape[0]
        epoch_losses = epoch_losses/n
        return tuple(epoch_losses.tolist())
    
    def valid_epoch(self) -> Tuple[float]:
        self.net.eval()
        epoch_losses = np.zeros((len(self.loss_names),), dtype=np.float32)
        n = 0
        for masks in tqdm(self.valid_loader):
            masks  =  masks.to(self.device)
            losses = self.valid_step(masks)
            n += masks.shape[0]
            epoch_losses += np.array(losses) * masks.shape[0]
        epoch_losses = epoch_losses/n
        return tuple(epoch_losses.tolist())
    
    def save_progress_image(self):
        raise NotImplementedError

    def train(
            self, 
            num_epochs      : int, 
            display_freq    : int = 5, 
            lambda_lr       : Union[Callable, None] = None,
        ):
        if lambda_lr is None:
            lambda_lr = lambda epoch : 1.0
        self.schedulers = [torch.optim.lr_scheduler.LambdaLR(optimizer, lambda_lr) 
                           for optimizer in self.optimizers]
        self.num_epochs = num_epochs
        self.nstep = 0
        
        for epoch in range(num_epochs):
            train_losses = self.train_epoch()
            valid_losses = self.valid_epoch()
            for scheduler in self.schedulers:
                scheduler.step()
            for i in range(len(self.loss_names)):
                self.train_losses[i].append(train_losses[i])
                self.valid_losses[i].append(valid_losses[i])
            msg = f"Epoch #{epoch}: "
            for i in range(len(self.loss_names)):
                msg = msg + f"{self.loss_names[i]}/train = {train_losses[i]:.5f}, "
            msg = msg[:-2] + " | "
            for i in range(len(self.loss_names)):
                msg = msg + f"{self.loss_names[i]}/valid = {valid_losses[i]:.5f}, "
            msg = msg[:-2]
            print(msg)
            if (epoch + 1) % display_freq == 0:
                self.save_progress_image(epoch)
            
class TrainerMaskVAE(TrainerMaskBase):
    loss_names : Tuple[str] = ("Rec_Loss", "KLD")
    def __init__(
            self, 
            net, 
            optimizer,
            kld_annealing_epochs,
            progress_dir,
            train_loader, 
            valid_loader,
            device = "cpu",
        ):
        super().__init__(net, train_loader, valid_loader, device)
        self.optimizer = optimizer
        self.optimizers = [self.optimizer]
        self.kld_annealing_steps = kld_annealing_epochs * len(train_loader)
        if kld_annealing_epochs == 0:
            self.get_kld_weight = lambda : 1.0
        else:
            self.get_kld_weight = lambda : torch.sigmoid(
                torch.tensor(self.nstep/self.kld_annealing_steps * 12 - 6)
            ).item()
        self.kld_loss_func = kld_loss
        self.rec_loss_func = dice_loss

        # Progress stuff
        self.progress_dir = progress_dir

        self.mask_fixed_t = next(iter(train_loader))
        self.mask_fixed_t = self.mask_fixed_t[:5].to(device)

        self.mask_fixed_v = next(iter(valid_loader))
        self.mask_fixed_v = self.mask_fixed_v[:5].to(device)

        self.z_fixed = get_noise(5, net.generator.z_dim, device)

    def train_step(self, masks: torch.Tensor) -> Tuple[float]:
        self.net.zero_grad()
        recons, mu, logvar = self.net(masks)
        recons = recons/2.0 + 0.5
        kld_loss = self.kld_loss_func(mu, logvar)
        rec_loss = self.rec_loss_func(masks, recons)
        (self.get_kld_weight() * kld_loss + rec_loss).backward()
        self.optimizer.step()
        return rec_loss.item(), kld_loss.item()
    
    def valid_step(self, masks: torch.Tensor) -> Tuple[float]:
    
        with torch.no_grad():
            recons, mu, logvar = self.net(masks)
            recons = recons/2.0 + 0.5
            kld_loss = self.kld_loss_func(mu, logvar)
            rec_loss = self.rec_loss_func(masks, recons)
            
        return rec_loss.item(), kld_loss.item()
    
    
    
    
    def save_progress_image(self, epoch):
        with torch.no_grad():
            recons_train = self.net(self.mask_fixed_t)[0]
            recons_train = recons_train/2.0 + 0.5 
            recons_valid = self.net(self.mask_fixed_v)[0]
            recons_valid = recons_valid/2.0 + 0.5
            generations  = self.net.generator(self.z_fixed)
            generations  = generations/2.0 + 0.5 

            img_grid = make_grid(
                torch.cat([
                    self.mask_fixed_t.cpu(), 
                    recons_train.cpu(),
                    self.mask_fixed_v.cpu(), 
                    recons_valid.cpu(), 
                    generations.cpu(),
                ]), 
                nrow=5, 
                padding=12, 
                pad_value=-1, 
            )
            plt.imsave(self.progress_dir / f"real_fake_{epoch+1:03d}.png", img_grid.numpy()[0] / 2.0 + 0.5)

    
