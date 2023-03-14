import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import make_grid

import numpy as np
import matplotlib.pyplot as plt

from networks import VAE, VAEGAN
from utils import kld_loss, get_noise
from typing import Tuple, Callable

from tqdm.auto import tqdm, trange

class TrainerBase():
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

    def train_step(self, images : torch.Tensor, masks : torch.Tensor) -> Tuple[float]:
        raise NotImplementedError
    
    def valid_step(self, images : torch.Tensor, masks : torch.Tensor) -> Tuple[float]:
        raise NotImplementedError

    def train_epoch(self) -> Tuple[float]:
        self.net.train()
        epoch_losses = np.zeros((len(self.loss_names),), dtype=np.float32)
        n = 0
        for images, masks in tqdm(self.train_loader):
            images = images.to(self.device)
            masks  =  masks.to(self.device)
            losses = self.train_step(images, masks)
            self.nstep += 1
            n += images.shape[0]
            epoch_losses += np.array(losses)
        epoch_losses = epoch_losses/n
        return tuple(epoch_losses.list())
    
    def valid_epoch(self) -> Tuple[float]:
        self.net.eval()
        epoch_losses = np.zeros((len(self.loss_names),), dtype=np.float32)
        n = 0
        for images, masks in tqdm(self.valid_loader):
            images = images.to(self.device)
            masks  =  masks.to(self.device)
            losses = self.valid_step(images, masks)
            n += images.shape[0]
            epoch_losses += np.array(losses)
        epoch_losses = epoch_losses/n
        return tuple(epoch_losses.list())
    
    def save_progress_image(self):
        raise NotImplementedError

    def train(
            self, 
            num_epochs      : int, 
            display_freq    : int = 5, 
            run_after_epoch : Callable = lambda trainer, epoch : None,
        ):
        self.num_epochs = num_epochs
        self.nstep = 0
        for epoch in range(num_epochs):
            train_losses = self.train_epoch()
            valid_losses = self.valid_epoch()
            run_after_epoch(self, epoch)
            msg = f"Epoch #{epoch}: "
            for i in range(len(self.loss_names)):
                msg = msg + f"{self.loss_names[i]}/train = {train_losses[i]:.5f}, "
            msg = msg[:-2] + " | "
            for i in range(len(self.loss_names)):
                msg = msg + f"{self.loss_names[i]}/valid = {valid_losses[i]:.5f}, "
            msg = msg[:-2]
            print(msg)
            if (epoch + 1) % display_freq == 0:
                self.save_progress_image()


class TrainerVAE(TrainerBase):
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
        self.kld_annealing_steps = kld_annealing_epochs * len(train_loader)
        if kld_annealing_epochs == 0:
            self.get_kld_weight = lambda : 1.0
        else:
            self.get_kld_weight = lambda : torch.sigmoid(
                torch.tensor(self.nstep/self.kld_annealing_steps * 12 - 6)
            ).item()
        self.kld_loss_func = kld_loss
        self.rec_loss_func = nn.L1Loss()

        # Progress stuff
        self.progress_dir = progress_dir

        self.x_fixed_t, self.y_fixed_t = next(iter(train_loader))[:5]
        self.x_fixed_t = self.x_fixed_t.to(device)
        self.y_fixed_t = self.y_fixed_t.to(device)

        self.x_fixed_v, self.y_fixed_v = next(iter(valid_loader))[:5]
        self.x_fixed_v = self.x_fixed_v.to(device)
        self.y_fixed_v = self.y_fixed_v.to(device)

        self.z_fixed = get_noise(5, net.encoder.z_dim, device)

    def train_step(self, images: torch.Tensor, masks: torch.Tensor) -> Tuple[float]:
        self.net.zero_grad()
        recons, mu, logvar = self.net(images, masks)
        kld_loss = self.kld_loss_func(mu, logvar)
        rec_loss = self.rec_loss_func(images, recons)
        (self.get_kld_weight() * kld_loss + rec_loss).backward()
        self.optimizer.step()
        return rec_loss.item(), kld_loss.item()
    
    def valid_step(self, images: torch.Tensor, masks: torch.Tensor) -> Tuple[float]:
        with torch.no_grad():
            recons, mu, logvar = self.net(images, masks)
            kld_loss = self.kld_loss_func(mu, logvar)
            rec_loss = self.rec_loss_func(images, recons)
        return rec_loss.item(), kld_loss.item()
    
    def save_progress_image(self):
        with torch.no_grad():
            recons_train = self.net(self.x_fixed_t, self.y_fixed_t)[0].cpu()
            recons_valid = self.net(self.x_fixed_v, self.y_fixed_v)[0].cpu()
            generations  = self.net.generator(self.z_fixed, self.y_fixed_v)

            img_grid = make_grid(
                torch.cat([
                    self.x_fixed_t, 
                    recons_train,
                    self.y_fixed_t, 
                    recons_valid, 
                    generations,
                ]), 
                nrow=5, 
                padding=12, 
                pad_value=-1, 
            )
            plt.imsave(self.progress_dir, img_grid.numpy()[0] / 2.0 + 0.5)

    
class TrainerVAEGAN(TrainerBase):
    loss_names : Tuple[str] = ("Rec_Loss", "KLD", "Discl_Loss", "Adv_Loss")
    def __init__(
            self, 
            net, 
            optimizer_enc,
            optimizer_gen,
            optimizer_disc,
            kld_annealing_epochs,
            progress_dir,
            train_loader, 
            valid_loader,
            gamma = 1.0,
            device = "cpu",
        ):
        super().__init__(net, train_loader, valid_loader, device)
        self.optimizer_enc  = optimizer_enc
        self.optimizer_gen  = optimizer_gen
        self.optimizer_disc = optimizer_disc
        self.kld_annealing_steps = kld_annealing_epochs * len(train_loader)
        if kld_annealing_epochs == 0:
            self.get_kld_weight = lambda : 1.0
        else:
            self.get_kld_weight = lambda : torch.sigmoid(
                torch.tensor(self.nstep/self.kld_annealing_steps * 12 - 6)
            ).item()
        self.kld_loss_func  = kld_loss
        self.rec_loss_func  = nn.L1Loss()
        self.discl_loss_func = nn.MSELoss()
        self.adv_loss_func  = nn.BCEWithLogitsLoss()
        self.gamma = gamma

        # Progress stuff
        self.progress_dir = progress_dir

        self.x_fixed_t, self.y_fixed_t = next(iter(train_loader))[:5]
        self.x_fixed_t = self.x_fixed_t.to(device)
        self.y_fixed_t = self.y_fixed_t.to(device)

        self.x_fixed_v, self.y_fixed_v = next(iter(valid_loader))[:5]
        self.x_fixed_v = self.x_fixed_v.to(device)
        self.y_fixed_v = self.y_fixed_v.to(device)

        self.z_fixed = get_noise(5, net.encoder.z_dim, device)

    def train_step(self, images: torch.Tensor, masks: torch.Tensor) -> Tuple[float]:
        recons, mu, logvar, scores_real, features_real, scores_fake, features_fake = self.net(images, masks)
        kld_loss  = self.kld_loss_func(mu, logvar)
        rec_loss  = self.rec_loss_func(images, recons)
        discl_loss = self.discl_loss_func(features_real, features_fake)
        adv_loss  = 0.5 * self.adv_loss_func(scores_real, torch.ones_like(scores_real)) + \
                    0.5 * self.adv_loss_func(scores_fake, torch.ones_like(scores_fake))
        
        # Encoder Step
        self.net.zero_grad()
        (self.get_kld_weight() * kld_loss + discl_loss).backward(retain_graph=True)
        self.optimizer_enc.step()

        # Generator Step
        self.net.zero_grad()
        (self.gamma * discl_loss - adv_loss).backward(retain_graph=True)
        self.optimizer_gen.step()

        # Discriminator Step
        self.net.discriminator.zero_grad()
        (adv_loss).backward()
        self.optimizer_disc.step()

        return rec_loss.item(), kld_loss.item(), discl_loss.item(), adv_loss.item()
    
    def valid_step(self, images: torch.Tensor, masks: torch.Tensor) -> Tuple[float]:
        with torch.no_grad():
            recons, mu, logvar, scores_real, features_real, scores_fake, features_fake = self.net(images, masks)
            kld_loss  = self.kld_loss_func(mu, logvar)
            rec_loss  = self.rec_loss_func(images, recons)
            discl_loss = self.discl_loss_func(features_real, features_fake)
            adv_loss  = 0.5 * self.adv_loss_func(scores_real, torch.ones_like(scores_real)) + \
                        0.5 * self.adv_loss_func(scores_fake, torch.ones_like(scores_fake))
        return rec_loss.item(), kld_loss.item(), discl_loss.item(), adv_loss.item()
    
    def save_progress_image(self):
        with torch.no_grad():
            recons_train = self.net(self.x_fixed_t, self.y_fixed_t)[0].cpu()
            recons_valid = self.net(self.x_fixed_v, self.y_fixed_v)[0].cpu()
            generations  = self.net.generator(self.z_fixed, self.y_fixed_v)

            img_grid = make_grid(
                torch.cat([
                    self.x_fixed_t, 
                    recons_train,
                    self.y_fixed_t, 
                    recons_valid, 
                    generations,
                ]), 
                nrow=5, 
                padding=12, 
                pad_value=-1, 
            )
            plt.imsave(self.progress_dir, img_grid.numpy()[0] / 2.0 + 0.5)