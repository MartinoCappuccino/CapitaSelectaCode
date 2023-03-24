import torch
import torch.nn as nn
from torchvision.utils import make_grid
import numpy as np
import matplotlib.pyplot as plt
from utils import kld_loss, get_noise
from typing import Tuple, Callable, List, Union
from utils import dice_loss, DiceBCELoss, kld_loss, get_noise, accumulate, get_kl_weight
from nonleaking import AdaptiveAugment
from tqdm.auto import tqdm
import os
from utils import sample_z

class TrainerBase():
    loss_names : Tuple[str] = ()
    def __init__(
        self,
        net,
        train_loader,
        valid_loader,
        CHECKPOINTS_DIR,
        TOLERANCE = 0.01,
        minimum_valid_loss = 10e6,
        device = "cpu",
        seed = 0,
        early_stopping = False,
    ):
        self.net = net
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.CHECKPOINTS_DIR = CHECKPOINTS_DIR
        self.minimum_valid_loss = minimum_valid_loss
        self.TOLERANCE = TOLERANCE
        self.early_stopping = early_stopping
        self.device = device

        self.optimizers : List[torch.optim.Optimizer] 
        self.train_losses = [[] for i in range(len(self.loss_names))]
        self.valid_losses = [[] for i in range(len(self.loss_names))]

        # Progress stuff
        np.random.seed(seed)

    def train_step(self, images : torch.Tensor, masks : torch.Tensor) -> Tuple[float]:
        raise NotImplementedError
    
    def valid_step(self, images : torch.Tensor, masks : torch.Tensor) -> Tuple[float]:
        raise NotImplementedError

    def train_epoch(self) -> Tuple[float]:
        self.net.train()
        epoch_losses = np.zeros((len(self.loss_names),), dtype=np.float32)
        n = 0
        for images, masks in tqdm(self.train_loader, leave=False):
            images = images.to(self.device)
            masks = masks.to(self.device)
            losses = self.train_step(images, masks)
            self.nstep += 1
            n += images.shape[0]
            epoch_losses += np.array(losses) * images.shape[0]
        epoch_losses = epoch_losses/n
        return tuple(epoch_losses.tolist())
    
    def valid_epoch(self) -> Tuple[float]:
        self.net.eval()
        epoch_losses = np.zeros((len(self.loss_names),), dtype=np.float32)
        n = 0
        for images, masks in tqdm(self.valid_loader, leave=False):
            images = images.to(self.device)
            masks  =  masks.to(self.device)
            losses = self.valid_step(images, masks)
            n += images.shape[0]
            epoch_losses += np.array(losses) * images.shape[0]
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
        self.nstep=0
        no_increase = 0
        for epoch in range(num_epochs):
            train_losses = self.train_epoch()
            valid_losses = self.valid_epoch()
            for scheduler in self.schedulers:
                scheduler.step()
            for i in range(len(self.loss_names)):
                self.train_losses[i].append(train_losses[i])
                self.valid_losses[i].append(valid_losses[i])
            msg = f"Epoch #{epoch:03d}: "
            for i in range(len(self.loss_names)):
                msg = msg + f"{self.loss_names[i]}/train = {train_losses[i]:.3f}, "
            msg = msg[:-2] + " | "
            for i in range(len(self.loss_names)):
                msg = msg + f"{self.loss_names[i]}/valid = {valid_losses[i]:.3f}, "
            msg = msg[:-2]
            tqdm.write(msg)
            if (epoch + 1) % display_freq == 0:
                self.save_progress_image(epoch)

            avg_losses = valid_losses[0]

            if (avg_losses) < self.minimum_valid_loss + self.TOLERANCE:
                no_increase = 0
                self.minimum_valid_loss = avg_losses
                if os.path.exists(self.CHECKPOINTS_DIR / f"model.pth"):
                    os.remove( self.CHECKPOINTS_DIR / f"model.pth")
                torch.save(
                    self.net.state_dict(),
                    self.CHECKPOINTS_DIR / f"model.pth",
                )
            else:
                no_increase +=1
                if self.early_stopping:
                    if no_increase > 9:
                        break

                    
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
            CHECKPOINTS_DIR,
            TOLERANCE = 0.01,
            minimum_valid_loss = 10e6,
            device = "cpu",
            seed = 0,
            early_stopping = False
        ):
        super().__init__(net, train_loader, valid_loader, CHECKPOINTS_DIR, TOLERANCE, minimum_valid_loss, device, seed, early_stopping)
        self.optimizer = optimizer
        self.optimizers = [self.optimizer]
        self.kld_annealing_steps = kld_annealing_epochs * len(train_loader)
        if kld_annealing_epochs == 0:
            self.get_kld_weight = lambda : 1.0
        else:
#             self.get_kld_weight = lambda : torch.sigmoid(
#                 torch.tensor(self.nstep/self.kld_annealing_steps * 12 - 6)
#             ).item()
            self.get_kld_weight = lambda : get_kl_weight(self.nstep, self.kld_annealing_steps)
        self.kld_loss_func = kld_loss
        self.rec_loss_func = nn.L1Loss()

        self.progress_dir = progress_dir

        np.random.seed(seed)

        indx_t = np.random.choice(np.arange(len(train_loader.dataset)), size=5, replace=False)
        self.x_fixed_t, self.y_fixed_t = train_loader.dataset[indx_t]
        self.x_fixed_t = self.x_fixed_t[:5].to(device)
        self.y_fixed_t = self.y_fixed_t[:5].to(device)

        indx_v = np.random.choice(np.arange(len(valid_loader.dataset)), size=5, replace=False)
        self.x_fixed_v, self.y_fixed_v = valid_loader.dataset[indx_v]
        self.x_fixed_v = self.x_fixed_v[:5].to(device)
        self.y_fixed_v = self.y_fixed_v[:5].to(device)

        self.z_fixed = get_noise(5, net.generator.z_dim, device)

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
    
    def save_progress_image(self, epoch):
        with torch.no_grad():
            recons_train = self.net(self.x_fixed_t, self.y_fixed_t)[0]
            recons_valid = self.net(self.x_fixed_v, self.y_fixed_v)[0]
            generations  = self.net.generator(self.z_fixed, self.y_fixed_v)

            img_grid = make_grid(
                torch.cat([
                    self.x_fixed_t.cpu(), 
                    recons_train.cpu(),
                    self.x_fixed_v.cpu(), 
                    recons_valid.cpu(), 
                    generations.cpu(),
                ]), 
                nrow=5, 
                padding=12, 
                pad_value=-1, 
            )
            plt.imsave(self.progress_dir / f"real_fake_{epoch+1:03d}.png", img_grid.numpy()[0] / 2.0 + 0.5)

    
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
            CHECKPOINTS_DIR,
            TOLERANCE = 0.01,
            minimum_valid_loss = 10e6,
            net_ema = None,
            accum = 0.999,
            gamma = 1.0,
            ada_target = 0,
            ada_length = 10000,
            device = "cpu",
            seed = 0,
            early_stopping = False
        ):
        super().__init__(net, train_loader, valid_loader, CHECKPOINTS_DIR, TOLERANCE, minimum_valid_loss, device, seed, early_stopping)
        self.optimizer_enc  = optimizer_enc
        self.optimizer_gen  = optimizer_gen
        self.optimizer_disc = optimizer_disc
        self.optimizers = [self.optimizer_enc, self.optimizer_gen, self.optimizer_disc]
        self.kld_annealing_steps = kld_annealing_epochs * len(train_loader)
        if kld_annealing_epochs == 0:
            self.get_kld_weight = lambda : 1.0
        else:
#             self.get_kld_weight = lambda : torch.sigmoid(
#                 torch.tensor(self.nstep/self.kld_annealing_steps * 12 - 6)
#             ).item()
            self.get_kld_weight = lambda : get_kl_weight(self.nstep, self.kld_annealing_steps)
        self.kld_loss_func   = kld_loss
        self.rec_loss_func   = nn.L1Loss()
        self.discl_loss_func = nn.MSELoss()
        self.adv_loss_func   = nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.progress_dir = progress_dir

        # ADA stuff
        self.ada_p = 0
        self.r_t_stat = 0
        self.ada_augment = AdaptiveAugment(ada_target, ada_length, 8, device)
        self.ada_p_log = []

        # EMA
        self.accum   = accum
        self.net_ema = net_ema

        np.random.seed(seed)

        indx_t = np.random.choice(np.arange(len(train_loader.dataset)), size=5, replace=False)
        indx_t_next = np.random.choice(np.arange(len(train_loader.dataset)), size=5, replace=False)
        
        self.x_fixed_t, self.y_fixed_t = train_loader.dataset[indx_t]
        self.x_fixed_t = self.x_fixed_t.to(device)
        self.y_fixed_t = self.y_fixed_t.to(device)
        
        self.x_next_t, self.y_next_t = train_loader.dataset[indx_t_next]
        self.x_next_t = self.x_next_t.to(device)
        self.y_next_t = self.y_next_t.to(device)

        indx_v = np.random.choice(np.arange(len(valid_loader.dataset)), size=5, replace=False)
        self.x_fixed_v, self.y_fixed_v = valid_loader.dataset[indx_v]
        self.x_fixed_v = self.x_fixed_v.to(device)
        self.y_fixed_v = self.y_fixed_v.to(device)

        self.z_fixed = get_noise(5, net.generator.z_dim, device)

    def train_step(self, images: torch.Tensor, masks: torch.Tensor) -> Tuple[float]:
        recons, mu, logvar, features_real, features_fake, scores_real, scores_fake = self.net(
            images, 
            masks, 
            ada_p=self.ada_p,
        )
        kld_loss  = self.kld_loss_func(mu, logvar)
        rec_loss  = self.rec_loss_func(images, recons)
        discl_loss = self.discl_loss_func(features_real, features_fake)
        adv_loss  = 0.5 * self.adv_loss_func(scores_real, torch.ones_like(scores_real)) + \
                    0.5 * self.adv_loss_func(scores_fake, torch.zeros_like(scores_fake))
        
        # Encoder Step
        self.net.zero_grad()
        (self.get_kld_weight() * kld_loss + discl_loss).backward(
            inputs=list(self.net.encoder.parameters()), 
            retain_graph=True
        )
        self.optimizer_enc.step()

        # Generator Step
        self.net.zero_grad()
        (self.gamma * discl_loss - adv_loss).backward(
            inputs=list(self.net.generator.parameters()),
            retain_graph=True
        )
        self.optimizer_gen.step()

        # Discriminator Step
        self.net.discriminator.zero_grad()
        (adv_loss).backward(
            inputs=list(self.net.discriminator.parameters()),
            retain_graph=False,
        )
        self.optimizer_disc.step()

        # Adjust ADA
        if self.ada_augment.ada_target != 0:
            self.ada_p_log.append(self.ada_p)
            self.ada_p = self.ada_augment.tune(scores_real)
            self.r_t_stat = self.ada_augment.r_t_stat

        # EMA
        if self.net_ema is not None:
            accumulate(self.net_ema, self.net, self.accum)

        return rec_loss.item(), kld_loss.item(), discl_loss.item(), adv_loss.item()
    
    def valid_step(self, images: torch.Tensor, masks: torch.Tensor) -> Tuple[float]:
        with torch.no_grad():
            recons, mu, logvar, features_real, features_fake, scores_real, scores_fake = self.net(images, masks)
            kld_loss  = self.kld_loss_func(mu, logvar)
            rec_loss  = self.rec_loss_func(images, recons)
            discl_loss = self.discl_loss_func(features_real, features_fake)
            adv_loss  = 0.5 * self.adv_loss_func(scores_real, torch.ones_like(scores_real)) + \
                        0.5 * self.adv_loss_func(scores_fake, torch.zeros_like(scores_fake))
            
        return rec_loss.item(), kld_loss.item(), discl_loss.item(), adv_loss.item()
    
    def save_progress_image(self, epoch):
        with torch.no_grad():
            net = self.net_ema if self.net_ema is not None else self.net
            net.eval()
            recons_train = net(self.x_fixed_t, self.y_fixed_t)[0]
            recons_valid = net(self.x_fixed_v, self.y_fixed_v)[0]
            
            mu_first, logvar_first = self.net.encoder(self.x_fixed_t)
            latent_z_first = sample_z(mu_first, logvar_first)
            
            mu_second, logvar_second = self.net.encoder(self.x_next_t)
            latent_z_second = sample_z(mu_second, logvar_second)
            
            gen_latent_z = latent_z_first + (0.5*(latent_z_second - latent_z_first))
            generations  = self.net.generator(gen_latent_z, self.y_fixed_v)
            #generations  = net.generator(self.z_fixed, self.y_fixed_v)

            img_grid = make_grid(
                torch.cat([
                    self.x_fixed_t.cpu(), 
                    recons_train.cpu(),
                    self.x_fixed_v.cpu(), 
                    recons_valid.cpu(), 
                    generations.cpu(),
                ]), 
                nrow=5, 
                padding=12, 
                pad_value=-1, 
            )
            plt.imsave(self.progress_dir / f"real_fake_{epoch+1:03d}.png", img_grid.numpy()[0] / 2.0 + 0.5)
            
class TrainerMaskVAE(TrainerBase):
    loss_names : Tuple[str] = ("Rec_Loss", "KLD")
    def __init__(
            self, 
            net, 
            optimizer,
            kld_annealing_epochs,
            progress_dir,
            train_loader, 
            valid_loader,
            CHECKPOINTS_DIR,
            TOLERANCE=0.01,
            minimum_valid_loss=10e8,
            device = "cpu",
            seed = 0,
            early_stopping = False
        ):
        super().__init__(net, train_loader, valid_loader, CHECKPOINTS_DIR, TOLERANCE, minimum_valid_loss, device, seed, early_stopping)
        self.optimizer = optimizer
        self.optimizers = [self.optimizer]
        self.kld_annealing_steps = kld_annealing_epochs * len(train_loader)
        if kld_annealing_epochs == 0:
            self.get_kld_weight = lambda : 1.0
        else:
#             self.get_kld_weight = lambda : torch.sigmoid(
#                 torch.tensor(self.nstep/self.kld_annealing_steps * 12 - 6)
#             ).item()
            self.get_kld_weight = lambda : get_kl_weight(self.nstep, self.kld_annealing_steps)
        self.kld_loss_func = kld_loss
        self.rec_loss_func = dice_loss

        # Progress stuff
        self.progress_dir = progress_dir

        np.random.seed(seed)

        indx_t = np.random.choice(np.arange(len(train_loader.dataset)), size=5, replace=False)
        indx_t_next = np.random.choice(np.arange(len(train_loader.dataset)), size=5, replace=False)
        
        self.x_fixed_t, self.y_fixed_t = train_loader.dataset[indx_t]
        self.x_fixed_t = self.x_fixed_t.to(device)
        self.y_fixed_t = self.y_fixed_t.to(device)
        
        self.x_next_t, self.y_next_t = train_loader.dataset[indx_t_next]
        self.x_next_t = self.x_next_t.to(device)
        self.y_next_t = self.y_next_t.to(device)

        indx_v = np.random.choice(np.arange(len(valid_loader.dataset)), size=5, replace=False)
        self.x_fixed_v, self.y_fixed_v = valid_loader.dataset[indx_v]
        self.x_fixed_v = self.x_fixed_v.to(device)
        self.y_fixed_v = self.y_fixed_v.to(device)

        self.z_fixed = get_noise(5, net.generator.z_dim, device)

    def train_step(self, images: torch.Tensor, masks: torch.Tensor) -> Tuple[float]:
        self.net.zero_grad()
        recons, mu, logvar = self.net(masks)
        recons = recons/2.0 + 0.5
        kld_loss = self.kld_loss_func(mu, logvar)
        rec_loss = self.rec_loss_func(masks, recons)
        (self.get_kld_weight() * kld_loss + rec_loss).backward()
        self.optimizer.step()
        return rec_loss.item(), kld_loss.item()
    
    def valid_step(self, images: torch.Tensor, masks: torch.Tensor) -> Tuple[float]:
        with torch.no_grad():
            recons, mu, logvar = self.net(masks)
            recons = recons/2.0 + 0.5
            kld_loss = self.kld_loss_func(mu, logvar)
            rec_loss = self.rec_loss_func(masks, recons)
            
        return rec_loss.item(), kld_loss.item()
     
    def save_progress_image(self, epoch):
        with torch.no_grad():
            recons_train = self.net(self.y_fixed_t)[0]
            recons_train = recons_train/2.0 + 0.5 
            recons_valid = self.net(self.y_fixed_v)[0]
            recons_valid = recons_valid/2.0 + 0.5
            
            mu_first, logvar_first = self.net.encoder(self.y_fixed_t)
            latent_z_first = sample_z(mu_first, logvar_first)
            
            mu_second, logvar_second = self.net.encoder(self.y_next_t)
            latent_z_second = sample_z(mu_second, logvar_second)
            
            gen_latent_z = latent_z_first + ((latent_z_second - latent_z_first)/2)
            generations  = self.net.generator(gen_latent_z)
            #generations  = self.net.generator(self.z_fixed)
            generations  = generations/2.0 + 0.5 

            img_grid = make_grid(
                torch.cat([
                    self.y_fixed_t.cpu(), 
                    recons_train.cpu(),
                    self.y_fixed_v.cpu(), 
                    recons_valid.cpu(), 
                    generations.cpu(),
                ]), 
                nrow=5, 
                padding=12, 
                pad_value=-1, 
            )
            plt.imsave(self.progress_dir / f"real_fake_{epoch+1:03d}.png", img_grid.numpy()[0] / 2.0 + 0.5)
            
class TrainerUNET(TrainerBase):
    loss_names : Tuple[str] = (["DiceBCELoss"])
    def __init__(
            self, 
            net, 
            optimizer,
            progress_dir,
            train_loader, 
            valid_loader,
            mask_generator,
            image_generator,
            TOLERANCE = 0.01,
            minimum_valid_loss = 10,
            Number_of_fake = 0,
            CHECKPOINTS_DIR = None,
            device = "cpu",
            seed = 0,
            early_stopping = True
        ):
        super().__init__(net, train_loader, valid_loader, CHECKPOINTS_DIR, TOLERANCE, minimum_valid_loss, device, seed, early_stopping)
        self.optimizer = optimizer
        self.optimizers = [self.optimizer]

        self.loss_function = DiceBCELoss()

        # Progress stuff
        self.progress_dir = progress_dir
        self.mask_generator = mask_generator
        self.image_generator = image_generator
        self.Number_of_fake = Number_of_fake

        np.random.seed(seed)

        indx_t = np.random.choice(np.arange(len(train_loader.dataset)), size=5, replace=False)
        self.x_fixed_t, self.y_fixed_t = train_loader.dataset[indx_t]
        self.x_fixed_t = self.x_fixed_t.to(device)
        self.y_fixed_t = self.y_fixed_t[:, 0:1].to(device)

        indx_v = np.random.choice(np.arange(len(valid_loader.dataset)), size=5, replace=False)
        self.x_fixed_v, self.y_fixed_v = valid_loader.dataset[indx_v]
        self.x_fixed_v = self.x_fixed_v.to(device)
        self.y_fixed_v = self.y_fixed_v[:, 0:1].to(device)

    def train_step(self, inputs: torch.Tensor, masks: torch.Tensor) -> Tuple[float]:
        self.net.zero_grad()
        if self.Number_of_fake > 0:
            random_tensors = get_noise(self.Number_of_fake, self.mask_generator.generator.z_dim, self.device)
            generated_masks = self.mask_generator.generator(random_tensors)
            generated_masks = generated_masks * (torch.rand(generated_masks.shape[0]) > 0.1)
            generated_masks = generated_masks[:, None].repeat_interleave(2, dim=1)
            generated_masks[:,1] = 1 - generated_masks[:,1]
            random_tensors = get_noise(self.Number_of_fake, self.image_generator.generator.z_dim, self.device)
            generated_images  = self.image_generator.generator(random_tensors, generated_masks)
            inputs = torch.cat((inputs, generated_images), dim=0)
            masks = torch.cat((masks, generated_masks), dim=0)

        outputs = self.net(inputs)
        masks = masks[:, 0:1]
        loss = self.loss_function(outputs, masks.float())
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def valid_step(self, inputs: torch.Tensor, masks: torch.Tensor) -> Tuple[float]:
        with torch.no_grad():
            outputs = self.net(inputs)
            masks = masks[:, 0:1]
            loss = self.loss_function(outputs, masks.float())        
        return loss.item()
    
    
    def save_progress_image(self, epoch):
        with torch.no_grad():
            recons_train = torch.round(torch.sigmoid(self.net(self.x_fixed_t)))
            recons_valid = torch.round(torch.sigmoid(self.net(self.x_fixed_v)))

            img_grid = make_grid(
                torch.cat([
                    self.y_fixed_t.cpu(), 
                    recons_train.cpu(),
                    self.y_fixed_v.cpu(), 
                    recons_valid.cpu(), 
                ]), 
                nrow=5, 
                padding=12, 
                pad_value=-1, 
            )
            plt.imsave(self.progress_dir / f"real_fake_{epoch+1:03d}.png", img_grid.numpy()[0] / 2.0 + 0.5)

    
