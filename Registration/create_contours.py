import SimpleITK as sitk
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import scipy

data_dir = r'C:\Users\marti\OneDrive - TU Eindhoven\Documenten\Master\Q3\Capita Selecta\Project\Data\p102'

prostaat = sitk.ReadImage(os.path.join(data_dir, "prostaat.mhd"))
prostaat = sitk.GetArrayFromImage(prostaat)


mr_bffe = sitk.ReadImage(os.path.join(data_dir, "mr_bffe.mhd"))
mr_bffe = sitk.GetArrayFromImage(mr_bffe)

im1 = prostaat[42,:,:]
im2 = prostaat[55,:,:]
im3 = prostaat[60,:,:]


im_bffe = mr_bffe[42,:,:]
im_bffe = np.float32(im_bffe)

plt.imshow(im1)
plt.contour(im1, colors=["red"])
plt.contour(im2, colors=["green"])
plt.contour(im3, colors=["blue"])

plt.savefig(os.path.join(data_dir,'contours.png'), dpi=200)

