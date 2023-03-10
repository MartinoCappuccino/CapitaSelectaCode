import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from tqdm import tqdm
import matplotlib.pyplot as plt

import utils
import vae

# to ensure reproducible training/validation split
random.seed(42)


# directorys with data and to store training checkpoints and logs
WORKING_DIR = Path(r"C:\Users\20182371\Documents\TUe\8DM20_CS_Medical_Imaging\DeepLearning_Project")
DATA_DIR = WORKING_DIR / "TrainingData"
CHECKPOINTS_DIR = WORKING_DIR / "vae_model_weights"
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
PROGRESS_DIR = WORKING_DIR / "progress"
TENSORBOARD_LOGDIR = WORKING_DIR / "vae_runs"

# training settings and hyperparameters
NO_VALIDATION_PATIENTS = 2
IMAGE_SIZE = [64, 64]
BATCH_SIZE = 32
N_EPOCHS = 200
DECAY_LR_AFTER = 50
LEARNING_RATE = 1e-3
DISPLAY_FREQ = 5

# dimension of VAE latent space
Z_DIM = 256


# function to reduce the
def lr_lambda(the_epoch):
    """Function for scheduling learning rate"""
    return (
        1.0
        if the_epoch < DECAY_LR_AFTER
        else 1 - float(the_epoch - DECAY_LR_AFTER) / (N_EPOCHS - DECAY_LR_AFTER)
    )


# find patient folders in training directory
# excluding hidden folders (start with .)
patients = [
    path
    for path in DATA_DIR.glob("*")
    if not any(part.startswith(".") for part in path.parts)
]
random.shuffle(patients)

# split in training/validation after shuffling
partition = {
    "train": patients[:-NO_VALIDATION_PATIENTS],
    "validation": patients[-NO_VALIDATION_PATIENTS:],
}

# load training data and create DataLoader with batching and shuffling
dataset = utils.ProstateMRDataset(partition["train"], IMAGE_SIZE)
dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True,
    pin_memory=True,
)

# load validation data
valid_dataset = utils.ProstateMRDataset(partition["validation"], IMAGE_SIZE)
valid_dataloader = DataLoader(
    valid_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True,
    pin_memory=True,
)

# initialise model, optimiser
vae_model = vae.VAE()# TODO 
optimizer = torch.optim.Adam(vae_model.parameters(), lr = LEARNING_RATE, betas=(0.0,0.9))# TODO 
# add a learning rate scheduler based on the lr_lambda function
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda) # TODO

x_real = next(iter(valid_dataloader))
# training loop
writer = SummaryWriter(log_dir=TENSORBOARD_LOGDIR)  # tensorboard summary
for epoch in range(N_EPOCHS):
    current_train_loss = 0.0
    current_valid_loss = 0.0
    
    # TODO 
    # training iterations
    for inputs, labels in tqdm(dataloader, position=0):
        # needed to zero gradients in each iterations
        optimizer.zero_grad()
        recons, mu, logvar = vae_model(inputs)  # forward pass
        loss = vae.vae_loss(inputs, recons, mu, logvar)
        loss.backward()  # backpropagate loss
        current_train_loss += loss.item()
        optimizer.step()  # update weights
        
    # evaluate validation loss
    with torch.no_grad():
        vae_model.eval()
        for inputs in tqdm(valid_dataloader, position=0):
            recons, mu, logvar = vae_model(inputs)   # forward pass
            loss = vae.vae_loss(inputs, recons, mu, logvar)
            current_valid_loss += loss.item()
        
        vae_model.train()
    # write to tensorboard log
    writer.add_scalar("Loss/train", current_train_loss / len(dataloader), epoch)
    writer.add_scalar(
        "Loss/validation", current_valid_loss / len(valid_dataloader), epoch
    )
    print(f"Epoch #{epoch} Loss/train {current_train_loss / len(dataloader):.5f} | Loss/validation {current_valid_loss / len(valid_dataloader):.5f}")
    scheduler.step() # step the learning step scheduler

    # save examples of real/fake images
    if (epoch + 1) % DISPLAY_FREQ == 0:
        x_recon = vae_model(x_real)[0]
        img_grid = make_grid(
            torch.cat((x_recon[:5], x_real[:5])), nrow=5, padding=12, pad_value=-1
        )
        img = np.clip(img_grid[0][np.newaxis], -1, 1) / 2 + 0.5
        writer.add_image(
            "Real_fake", img, epoch + 1,
        )
        plt.imsave(PROGRESS_DIR / f"Real_fake_{epoch:03d}.png", img[0])
    
    # TODO: sample noise  
    # TODO: generate images and display NEED TO BE ADDED

torch.save(vae_model.state_dict(), CHECKPOINTS_DIR / "vae_model.pth")
