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



data_dir = r'C:\Users\20192024\OneDrive - TU Eindhoven\Documents\Y4\Q3\Capita Selecta in Medical Image Analysis\Project\Nieuw\TrainingData\p102'

prostaat = sitk.ReadImage(os.path.join(data_dir, "prostaat.mhd"))
prostaat = sitk.GetArrayFromImage(prostaat)


mr_bffe = sitk.ReadImage(os.path.join(data_dir, "mr_bffe.mhd"))
mr_bffe = sitk.GetArrayFromImage(mr_bffe)

im1 = prostaat[42,:,:]
im2 = prostaat[55,:,:]


im_bffe = mr_bffe[42,:,:]
im_bffe = np.float32(im_bffe)


np.array_equal(im1, im1.astype(bool))
plt.imshow(im1,cmap='gray')

contours1, hierarchy1 = cv.findContours(im1, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
contours2, hierarchy2 = cv.findContours(im2, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

#im2, contours = cv.findContours(im,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
color1 = cv.cvtColor(im1, cv.COLOR_GRAY2RGBA)
color2 = cv.cvtColor(im2, cv.COLOR_GRAY2RGBA)

plt.imshow(color1)
plt.imshow(color2)

cv.drawContours(color1, contours1, -1, (0,0,255), 2)
cv.drawContours(color2, contours2, -1, (0,255,0), 2)




#final = cv.cvtColor(color, cv.COLOR_BGR2GRAY)


cv.imshow('Contours', color1)
cv.imshow('Contours', color2)

cv.waitKey(0)


temp = np.zeros((im1.shape[0],im1.shape[1],3), dtype = np.uint8)
cv.drawContours(temp,contours1, -1,(0,0,255),1)
cv.imshow('Original Image',im1)
cv.imshow('Contours Drawn',temp)
cv.waitKey(0)
cv.destroyAllWindows()



cv.imwrite(os.path.join(data_dir, 'contours_none_image1.png'), color1)
cv.imwrite(os.path.join(data_dir, 'contours_none_image2.png'), color2)
cv.imwrite(os.path.join(data_dir, 'prostaat.png'), im1)

#cv.imwrite(os.path.join(data_dir, 'contours_none_image2.jpg'), temp)

cv.destroyAllWindows()

plt.imshow(color1)
plt.imshow(color2)




