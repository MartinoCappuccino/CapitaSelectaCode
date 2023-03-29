import random
from pathlib import Path

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

import u_net
import utils

device = "cuda" if torch.cuda.is_available() else "cpu"

# to ensure reproducible training/validation split
random.seed(42)

# directorys with data and to stored training checkpoints
DATA_DIR = Path.cwd() / "TrainingData"

# this is my best epoch - what is yours?
BEST_EPOCH = 30
CHECKPOINTS_DIR = Path.cwd() / "200_epochs_24_number_of_fake" / f"model.pth"

# hyperparameters
NO_VALIDATION_PATIENTS = 2
IMAGE_SIZE = [64, 64]

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

# load validation data
valid_dataset = utils.ProstateMRDataset(partition["validation"], IMAGE_SIZE)

unet_model = u_net.UNet(num_classes=1)
unet_model.load_state_dict(torch.load(CHECKPOINTS_DIR))
unet_model.eval()

# TODO
# apply for all images and compute Dice score with ground-truth.
# output .mhd images with the predicted segmentations
with torch.no_grad():
    predict_index = 75
    (input, target) = valid_dataset[predict_index]
    output = torch.sigmoid(unet_model(input[np.newaxis, ...]))
    prediction = torch.round(output)

    fig, ax = plt.subplots(1, 3)
    ax[0].imshow(input[0], cmap="gray")
    ax[0].set_title("Input")
    ax[0].axis("off")

    ax[1].imshow(target[0])
    ax[1].set_title("Ground-truth")
    ax[1].axis("off")

    ax[2].imshow(prediction[0, 0])
    ax[2].set_title("Prediction")
    ax[2].axis("off")
    plt.show()