import random
from pathlib import Path

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

import u_net
import utils

# to ensure reproducible training/validation split
random.seed(42)

# directorys with data and to stored training checkpoints
DATA_DIR = Path.cwd() / "TrainingData"

# this is my best epoch - what is yours?
#BEST_EPOCH = 30
CHECKPOINTS_DIR_0 = Path.cwd() / "200_epochs_0_number_of_fake" / f"model.pth"
CHECKPOINTS_DIR_4 = Path.cwd() / "200_epochs_4_number_of_fake" / f"model.pth"
CHECKPOINTS_DIR_8 = Path.cwd() / "200_epochs_8_number_of_fake" / f"model.pth"
CHECKPOINTS_DIR_24 = Path.cwd() / "200_epochs_24_number_of_fake" / f"model.pth"

# hyperparameters
#NO_VALIDATION_PATIENTS = 2
IMAGE_SIZE = [64, 64]

# find patient folders in training directory
# excluding hidden folders (start with .)
patients = [
    path
    for path in DATA_DIR.glob("*")
    if not any(part.startswith(".") for part in path.parts)
]
#random.shuffle(patients)

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
        
# print(train_split)
# print(validation_split)

# split in training/validation after shuffling
partition = {
    "train": train_split,
    "validation": validation_split,
}

# load validation data
valid_dataset = utils.ProstateMRDataset(partition["validation"], IMAGE_SIZE)

unet_model_0 = u_net.UNet(num_classes=1)
unet_model_0.load_state_dict(torch.load(CHECKPOINTS_DIR_0))
unet_model_0.eval()

unet_model_4 = u_net.UNet(num_classes=1)
unet_model_4.load_state_dict(torch.load(CHECKPOINTS_DIR_4))
unet_model_4.eval()

unet_model_8 = u_net.UNet(num_classes=1)
unet_model_8.load_state_dict(torch.load(CHECKPOINTS_DIR_8))
unet_model_8.eval()

unet_model_24 = u_net.UNet(num_classes=1)
unet_model_24.load_state_dict(torch.load(CHECKPOINTS_DIR_24))
unet_model_24.eval()

# TODO
# apply for all images and compute Dice score with ground-truth.
# output .mhd images with the predicted segmentations
# with torch.no_grad():
    # predict_index = 75
    # (input, target) = valid_dataset[predict_index]
    # output = torch.sigmoid(unet_model(input[np.newaxis, ...]))
    # prediction = torch.round(output)

    # fig, ax = plt.subplots(1, 3)
    # ax[0].imshow(input[0], cmap="gray")
    # ax[0].set_title("Input")
    # ax[0].axis("off")

    # ax[1].imshow(target[0])
    # ax[1].set_title("Ground-truth")
    # ax[1].axis("off")

    # ax[2].imshow(prediction[0, 0])
    # ax[2].set_title("Prediction")
    # ax[2].axis("off")
    # plt.show()
 

# Contours 

slice_list = [10,30,50,70]


fig, axs = plt.subplots(nrows=1, ncols=4, figsize=(10, 7))
plt.subplots_adjust(wspace =0.1 , hspace=0.1)
fig.suptitle("Contours for different slices of validation", fontsize=18, y=0.8)

for slic, ax in zip(range(len(slice_list)), axs.ravel()):
    slice_nr = slice_list[slic]
    
    with torch.no_grad():
        predict_index = slice_nr
        (input, target) = valid_dataset[predict_index]
        
        output_0 = torch.sigmoid(unet_model_0(input[np.newaxis, ...]))
        prediction_0 = torch.round(output_0)
        
        output_4 = torch.sigmoid(unet_model_4(input[np.newaxis, ...]))
        prediction_4 = torch.round(output_4)
        
        output_8 = torch.sigmoid(unet_model_8(input[np.newaxis, ...]))
        prediction_8 = torch.round(output_8)
        
        output_24 = torch.sigmoid(unet_model_24(input[np.newaxis, ...]))
        prediction_24 = torch.round(output_24)
        
    gt_1 = target[0]
    
    ax.imshow(gt_1, cmap='gray')
    c1 = ax.contour(prediction_0[0,0], colors=['red'], linewidths=1)
    c2 = ax.contour(prediction_4[0,0], colors=['royalblue'], linewidths=1)
    c3 = ax.contour(prediction_8[0,0], colors=['limegreen'], linewidths=1)
    c4 = ax.contour(prediction_24[0,0], colors=['magenta'], linewidths=1)
    
    if slice_nr == 10:
        h1,l1 = c1.legend_elements()
        h2,l1 = c2.legend_elements()
        h3,l1 = c3.legend_elements()
        h4,l1 = c4.legend_elements()
        
        ax.legend([h1[0], h2[0], h3[0], h4[0]], ['0 synthetic', '4 synthetic', '8 synthetic', '24 synthetic'], loc='lower left')


for i, ax in enumerate(fig.axes):
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.axis('off')
    ax.set_title('Slice ' + str(slice_list[i]), fontsize=16) 
    

plt.show()
##fig.savefig(os.path.join(data_dir, 'results', experiment, val_patient, 'plot_contours.png'), dpi=200)
