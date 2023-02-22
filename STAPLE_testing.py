# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 09:55:12 2023

@author: 20182371
"""

import SimpleITK as sitk
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

"""
This file can used to create one mask out of several 
masks using STAPLE algorithm. The mask should be saved 
as .mhd files in single folder. the path to that folder 
should be given to the variable maks_dir. 
"""

masks_dir = Path(r"C:\Users\20182371\Documents\TUe\8DM20_CS_Medical_Imaging\TrainingData\results")

segmentations = [x for x in masks_dir.glob("*.mhd")]
print(segmentations)

seg_stack = []
STAPLE_3D_seg = np.zeros((333, 271, 85))

for i in range(0, 85):
    for seg in segmentations:
        image = sitk.ReadImage(str(seg))
        image_array = sitk.GetArrayFromImage(image)
        seg_sitk = sitk.GetImageFromArray(image_array[i,:,:].astype(np.int16))
        seg_stack.append(seg_sitk)

    STAPLE_seg_sitk = sitk.STAPLE(seg_stack, 1.0 )
    STAPLE_seg = sitk.GetArrayFromImage(STAPLE_seg_sitk)
    STAPLE_3D_seg[:, :, i] = STAPLE_seg

#STAPLE_3D_seg = np.stack(image_stack, axis=0)
print(STAPLE_3D_seg.shape)
plt.imshow(STAPLE_3D_seg[50,:,:])







