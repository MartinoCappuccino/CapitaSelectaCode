import random
from pathlib import Path
import os

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import torch.nn.functional

import u_net
import utils

device = "cuda" if torch.cuda.is_available() else "cpu"

# to ensure reproducible training/validation split
random.seed(42)

# directorys with data and to stored training checkpoints
DATA_DIR = Path.cwd() / "TrainingData"

# load models
CHECKPOINTS_DIR_0 = Path.cwd() / "60_epochs_0_number_of_fake" / f"model.pth"
CHECKPOINTS_DIR_12 = Path.cwd() / "60_epochs_12_number_of_fake" / f"model.pth"
CHECKPOINTS_DIR_28 = Path.cwd() / "60_epochs_28_number_of_fake" / f"model.pth"

# hyperparameters
IMAGE_SIZE = [64, 64]

# find patient folders in training directory
# excluding hidden folders (start with .)
patients = [
    path
    for path in DATA_DIR.glob("*")
    if not any(part.startswith(".") for part in path.parts)
]

# Choose training and validation splits
train_split = []
validation_split = []

for i in patients:
    if 'p107' in str(i):
        validation_split.append(i)
    elif 'p117' in str(i): 
        validation_split.append(i)
    elif 'p120' in str(i):
        validation_split.append(i)
    else:
        train_split.append(i)

# split in training/validation after shuffling
partition = {
    "train": train_split,
    "validation": validation_split,
}

# load validation data
valid_dataset = utils.ProstateMRDataset(partition["validation"], IMAGE_SIZE)
valid_dataloader = DataLoader(valid_dataset, batch_size=1)

#models
unet_model_0 = u_net.UNet(num_classes=1).to(device)
unet_model_0.load_state_dict(torch.load(CHECKPOINTS_DIR_0, map_location=device))
unet_model_0.eval()

unet_model_12 = u_net.UNet(num_classes=1).to(device)
unet_model_12.load_state_dict(torch.load(CHECKPOINTS_DIR_12, map_location=device))
unet_model_12.eval()

unet_model_28 = u_net.UNet(num_classes=1).to(device)
unet_model_28.load_state_dict(torch.load(CHECKPOINTS_DIR_28, map_location=device))
unet_model_28.eval()

predictions_0 = []
predictions_12 = []
predictions_28 = [] 
targets = []
 
with torch.no_grad():
    for image, target in tqdm(valid_dataloader):
        image = image.to(device)
        target = target[:,0:1].to(device)
        
        #0 synthetic
        output_0 = torch.sigmoid(unet_model_0(image))
        prediction_0 = torch.round(output_0)

        prediction_0 = torch.nn.functional.interpolate(prediction_0, size=(333, 271), mode="nearest")
        target = torch.nn.functional.interpolate(target, size=(333, 271), mode="nearest")

        predictions_0.append(prediction_0)
        targets.append(target)
        
        #12 synthetic
        output_12 = torch.sigmoid(unet_model_12(image))
        prediction_12 = torch.round(output_12)

        prediction_12 = torch.nn.functional.interpolate(prediction_12, size=(333, 271), mode="nearest")
        
        predictions_12.append(prediction_12)
        
        #28 synthetic
        output_28 = torch.sigmoid(unet_model_28(image))
        prediction_28 = torch.round(output_28)

        prediction_28 = torch.nn.functional.interpolate(prediction_28, size=(333, 271), mode="nearest")
        
        predictions_28.append(prediction_28)
 
#----------------------------------------------------------------------
# Contours 

slice_list = [10,30,50,70]


fig, axs = plt.subplots(nrows=1, ncols=4, figsize=(10, 7))
plt.subplots_adjust(wspace =0.1 , hspace=0.1)
fig.suptitle("Contours for different slices of validation patient p107", fontsize=18, y=0.8)

for slic, ax in zip(range(len(slice_list)), axs.ravel()):
    slice_nr = slice_list[slic]
        
    gt_1 = targets[slice_nr][0,0]
    
    ax.imshow(gt_1, cmap='gray')
    c1 = ax.contour(predictions_0[slice_nr][0,0], colors=['red'], linewidths=1.1)
    c2 = ax.contour(predictions_12[slice_nr][0,0], colors=['royalblue'], linewidths=1.1)
    c3 = ax.contour(predictions_28[slice_nr][0,0], colors=['limegreen'], linewidths=1.1)
    
    if slice_nr == 10:
        h1,l1 = c1.legend_elements()
        h2,l1 = c2.legend_elements()
        h3,l1 = c3.legend_elements()
        
        ax.legend([h1[0], h2[0], h3[0]], ['0 synthetic', '12 synthetic', '28 synthetic'], loc='lower left')


for i, ax in enumerate(fig.axes):
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.axis('off')
    ax.set_title('Slice ' + str(slice_list[i]), fontsize=16) 
    

plt.show()
fig.savefig(os.path.join(Path.cwd(), 'plot_contours_ML.png'), dpi=200)
