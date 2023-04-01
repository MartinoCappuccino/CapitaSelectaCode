import SimpleITK as sitk
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import scipy
from skimage import metrics


data_dir = r'C:\\Users\\20192024\\OneDrive - TU Eindhoven\\Documents\\Y4\\Q3\\Capita Selecta in Medical Image Analysis\\Project\\Nieuw\\'
val_patient = 'p107'
at1 = 'p102'
at2 = 'p133'
experiment = 'TransAffineBSpline04'


gt = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(data_dir,'TrainingData', val_patient, 'prostaat.mhd')))
atlas1= sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(data_dir,'results', experiment, val_patient, at1, 'result.mhd')))
atlas2= sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(data_dir,'results', experiment, val_patient, at2, 'result.mhd')))
staple= sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(data_dir,'results', experiment, val_patient, 'staple.mhd')))

# Show / compare results for different slices of the prostate

slice_list = [10,30,50,70]


fig, axs = plt.subplots(nrows=1, ncols=4, figsize=(10, 7))
plt.subplots_adjust(wspace =0.1 , hspace=0.1)
fig.suptitle("Contours for different slices of validation patient p107", fontsize=18, y=0.8)

for slic, ax in zip(range(len(slice_list)), axs.ravel()):
    slice_nr = slice_list[slic]
    gt_1 = gt[slice_nr,:,:]

    # Have to convert the results back to uint8 because we changed it to float32 because of the ripples
    im1 = cv.normalize(atlas1[slice_nr,:,:], None, 255, 0, cv.NORM_MINMAX, cv.CV_8U)
    im1 = np.where(im1>128, 255, 0)
    im2 = cv.normalize(atlas2[slice_nr,:,:], None, 255, 0, cv.NORM_MINMAX, cv.CV_8U)
    im2 = np.where(im2>128, 255, 0)
    im3 = staple[slice_nr,:,:]
    
    ax.imshow(gt_1, cmap='gray')
    c1 = ax.contour(im1, colors=['red'], linewidths=1.1)
    c2 = ax.contour(im2, colors=['royalblue'], linewidths=1.1)
    c3 = ax.contour(im3, colors=['limegreen'], linewidths=1.1)
    
    if slice_nr == 10:
        h1,l1 = c1.legend_elements()
        h2,l1 = c2.legend_elements()
        h3,l1 = c3.legend_elements()
        
        ax.legend([h1[0], h2[0], h3[0]], ['atlas1', 'atlas2', 'STAPLE'], loc='lower left')


for i, ax in enumerate(fig.axes):
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.axis('off')
    ax.set_title('Slice ' + str(slice_list[i]), fontsize=16) 
    

plt.show()
fig.savefig(os.path.join(data_dir, 'results', experiment, val_patient, 'plot_contours.png'), dpi=200)


# =============================================================================
# Debugging HD
# =============================================================================

from scipy.spatial.distance import directed_hausdorff

example_gt1 = gt[50,:,:]
example_at1 = atlas1[50,:,:]

example_gt2 = gt[60,:,:]
example_at2 = atlas1[60,:,:]

hausdorf_list = [directed_hausdorff(example_gt1, example_at1), directed_hausdorff(example_gt2, example_at2)]

hausdorf = np.array(hausdorf_list)
mean_hausdorf = hausdorf.mean()
std_hausdorf = hausdorf.std()
















