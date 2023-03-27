# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 10:09:06 2023

@author: 20192024
"""
import SimpleITK as sitk
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import scipy

# Show contours for the validation patient 107

data_dir = r'C:\\Users\\20192024\\OneDrive - TU Eindhoven\\Documents\\Y4\\Q3\\Capita Selecta in Medical Image Analysis\\Project\\Nieuw\\'


gt = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(data_dir,'TrainingData\\p107\\prostaat.mhd')))
atlas1= sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(data_dir,'results\\TransAffineBSpline04\\p107\\p102\\result.mhd')))
atlas2= sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(data_dir,'results\\TransAffineBSpline04\\p107\\p133\\result.mhd')))
staple= sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(data_dir,'results\\TransAffineBSpline04\\p107\\staple.mhd')))

# Show / compare results for different slices of the prostate

slice_list = [10,30,50,70]
#slice_list = [10,15,20,25]

# SLICE 10

slice_nr = slice_list[0]
gt_1 = gt[slice_nr,:,:]

# Have to convert the results back to uint8 because we changed it to float32 because of the ripples
im1 = cv.normalize(atlas1[slice_nr,:,:], None, 255, 0, cv.NORM_MINMAX, cv.CV_8U)
im1 = np.where(im1>128, 255, 0)
im2 = cv.normalize(atlas2[slice_nr,:,:], None, 255, 0, cv.NORM_MINMAX, cv.CV_8U)
im2 = np.where(im2>128, 255, 0)
im3 = staple[slice_nr,:,:]

#SLICE 30

slice_nr = slice_list[1]
gt_2 = gt[slice_nr,:,:]

# Have to convert the results back to uint8 because we changed it to float32 because of the ripples
im1_2 = cv.normalize(atlas1[slice_nr,:,:], None, 255, 0, cv.NORM_MINMAX, cv.CV_8U)
im1_2 = np.where(im1_2>128, 255, 0)
im2_2 = cv.normalize(atlas2[slice_nr,:,:], None, 255, 0, cv.NORM_MINMAX, cv.CV_8U)
im2_2 = np.where(im2_2>128, 255, 0)
im3_2 = staple[slice_nr,:,:]



#SLICE 50
slice_nr = slice_list[2]
gt_3 = gt[slice_nr,:,:]

# Have to convert the results back to uint8 because we changed it to float32 because of the ripples
im1_3 = cv.normalize(atlas1[slice_nr,:,:], None, 255, 0, cv.NORM_MINMAX, cv.CV_8U)
im1_3 = np.where(im1_3>128, 255, 0)
im2_3 = cv.normalize(atlas2[slice_nr,:,:], None, 255, 0, cv.NORM_MINMAX, cv.CV_8U)
im2_3 = np.where(im2_3>128, 255, 0)
im3_3 = staple[slice_nr,:,:]


#SLICE 70
slice_nr = slice_list[3]
gt_4 = gt[slice_nr,:,:]

# Have to convert the results back to uint8 because we changed it to float32 because of the ripples
im1_4 = cv.normalize(atlas1[slice_nr,:,:], None, 255, 0, cv.NORM_MINMAX, cv.CV_8U)
im1_4 = np.where(im1_4>128, 255, 0)
im2_4 = cv.normalize(atlas2[slice_nr,:,:], None, 255, 0, cv.NORM_MINMAX, cv.CV_8U)
im2_4 = np.where(im2_4>128, 255, 0)
im3_4 = staple[slice_nr,:,:]


# Plotting everything

fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(10, 7))
plt.subplots_adjust(wspace =0.1 , hspace=0.1)
fig.suptitle("Contours for different slices of validation patient p107", fontsize=18, y=0.8)



ax[0].imshow(gt_1, cmap='gray')
c1 = ax[0].contour(im1, colors=['red'], linewidths=1.1)
c2 = ax[0].contour(im2, colors=['royalblue'], linewidths=1.1)
c3 = ax[0].contour(im3, colors=['limegreen'], linewidths=1.1)
h1,l1 = c1.legend_elements()
h2,l1 = c2.legend_elements()
h3,l1 = c3.legend_elements()
fig.legend([h1[0], h2[0], h3[0]], ['atlas1', 'atlas2', 'STAPLE'])



ax[1].imshow(gt_2, cmap='gray')
c1 = ax[1].contour(im1_2, colors=['red'], linewidths=1.1)
c2 = ax[1].contour(im2_2, colors=['royalblue'], linewidths=1.1)
c3 = ax[1].contour(im3_2, colors=['limegreen'], linewidths=1.1)




ax[2].imshow(gt_3, cmap='gray')
c1 = ax[2].contour(im1_3, colors=['red'], linewidths=1.1)
c2 = ax[2].contour(im2_3, colors=['royalblue'], linewidths=1.1)
c3 = ax[2].contour(im3_3, colors=['limegreen'], linewidths=1.1)



ax[3].imshow(gt_4, cmap='gray')
c1 = ax[3].contour(im1_4, colors=['red'], linewidths=1.1)
c2 = ax[3].contour(im2_4, colors=['royalblue'], linewidths=1.1)
c3 = ax[3].contour(im3_4, colors=['limegreen'], linewidths=1.1)

for i, ax in enumerate(fig.axes):
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.axis('off')
    ax.set_title('Slice ' + str(slice_list[i]), fontsize=16)



plt.show()


fig.savefig(os.path.join(data_dir, 'TrainingData\\p107\\', 'slice'+str(slice_nr)+'_contours.png'), dpi=200)


# contours1, hierarchy1 = cv.findContours(im1, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
# contours2, hierarchy2 = cv.findContours(im2, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
# contours3, hierarchy3 = cv.findContours(im3, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)


# #im2, contours = cv.findContours(im,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
# color1 = cv.cvtColor(im1, cv.COLOR_GRAY2RGB)
# color2 = cv.cvtColor(im2, cv.COLOR_GRAY2RGB)
# color3 = cv.cvtColor(im2, cv.COLOR_GRAY2RGB)


# plt.imshow(color1)
# plt.imshow(color2)

# cv.drawContours(color1, contours1, -1, (0,255,0), 2)
# cv.drawContours(color2, contours2, -1, (255,0,0), 2)
# cv.drawContours(color3, contours3, -1, (0,0,255), 2)

# plt.imshow(im1, cmap='gray')
# plt.imshow(color2, alpha=0.4)
# plt.imshow(color1, alpha=0.4)
# plt.imshow(color3, alpha=0.5)

# plt.savefig(os.path.join(data_dir,str(slice_nr),'_contours.png'), dpi=200)

# attempt to make the background transparent but not really working :(

# tmp = cv.cvtColor(color2, cv.COLOR_BGR2GRAY)
# _,alpha = cv.threshold(tmp,0,255,cv.THRESH_BINARY)
# b, g, r = cv.split(color2)
# rgba = [b,g,r, alpha]
# dst = cv.merge(rgba,4)
# cv.imwrite(os.path.join(data_dir,"test.png"), dst)

# =============================================================================
# left overs
# =============================================================================
#temp = np.zeros((im1.shape[0],im1.shape[1],3), dtype = np.uint8)

# b = im1
# temp = np.repeat(b[:, :, np.newaxis], 3, axis=2)

# cv.drawContours(temp,contours2, -1,(0,0,255),1)
# cv.imshow('Original Image',im1)
# cv.imshow('Contours Drawn',temp)
# cv.waitKey(0)
# cv.destroyAllWindows()



# cv.imwrite(os.path.join(data_dir, 'contours_none_image1.png'), color1)
# cv.imwrite(os.path.join(data_dir, 'contours_none_image2.png'), temp)
# cv.imwrite(os.path.join(data_dir, 'prostaat.png'), im1)

# #cv.imwrite(os.path.join(data_dir, 'contours_none_image2.jpg'), temp)

# cv.destroyAllWindows()


