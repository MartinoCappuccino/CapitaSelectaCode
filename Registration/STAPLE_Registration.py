# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 09:55:12 2023

@author: 20182371
"""

from __future__ import print_function, absolute_import
import elastix
import matplotlib.pyplot as plt
import imageio
import os
import SimpleITK as sitk
from pathlib import Path
import numpy as np
from IndexTracker import IndexTracker
import pandas as pd


"""
This file can used to create one mask out of several 
masks using STAPLE algorithm. The mask should be saved 
as .mhd files in single folder. the path to that folder 
should be given to the variable maks_dir. 

source: https://towardsdatascience.com/how-to-use-the-staple-algorithm-to-combine-multiple-image-segmentations-ce91ebeb451e
"""

def staple(data_path, fixed_patient):
    
    result_dir = os.path.join(data_path, "results", fixed_patient)
    if os.path.exists(result_dir) is False:
        raise IOError('results cannot be found')
    
    patients = os.listdir(result_dir)

    seg_stack = []
    STAPLE_3D_seg = np.zeros((86,333, 271))

    for i in range(0, 85):
        for patient in patients:
            if patient[0] == "p" and patient != fixed_patient:
                seg_path = os.path.join(result_dir, patient, "prostaat", "result.mhd" )
                image = sitk.ReadImage(str(seg_path))
                image_array = sitk.GetArrayFromImage(image)
                seg_sitk = sitk.GetImageFromArray(image_array[i,:,:].astype(np.int16))
                seg_stack.append(seg_sitk)
        
        STAPLE_seg_sitk = sitk.STAPLE(seg_stack, 1.0 )
        STAPLE_seg = sitk.GetArrayFromImage(STAPLE_seg_sitk)
        STAPLE_3D_seg[i, :, :] = STAPLE_seg

    #STAPLE_3D_seg = np.stack(image_stack, axis=0)
    STAPLE_3D_seg = np.where(STAPLE_3D_seg > 0.95, 1, 0)
    STAPLE_3D_seg = np.nan_to_num(STAPLE_3D_seg, nan=0)
    
    return STAPLE_3D_seg

def staple_3D(data_path, fixed_patient):
    
    result_dir = os.path.join(data_path, "results", fixed_patient)
    if os.path.exists(result_dir) is False:
        raise IOError('results cannot be found')
    
    patients = os.listdir(result_dir)

    seg_stack = []

    for patient in patients:
        if patient[0] == "p" and patient != fixed_patient:
            seg_path = os.path.join(result_dir, patient, "prostaat", "result.mhd" )
            image = sitk.ReadImage(str(seg_path))
            image_array = sitk.GetArrayFromImage(image)
            seg_sitk = sitk.GetImageFromArray(image_array.astype(np.int16))
            seg_stack.append(seg_sitk)
        
        
    STAPLE_seg_sitk = sitk.STAPLE(seg_stack, 1.0 )
    STAPLE_seg = sitk.GetArrayFromImage(STAPLE_seg_sitk)

    #STAPLE_3D_seg = np.stack(image_stack, axis=0)
    STAPLE_seg = np.where(STAPLE_seg > 0.95, 1, 0)
    STAPLE_seg = np.nan_to_num(STAPLE_seg, nan=0)
    
    #sitk.WriteImage(STAPLE_seg, os.path.join(result_dir, "STAPLE_seg.nii"))
    
    return STAPLE_seg

def registration_transformation(elastix_path, data_path, fixed_patient, show_results = False):
    """ This function registrates all the images in de datapath that are not the fixed 
    image to the fixed image and then tranforms the labels of the moving images to 
    resemble that of the fixed image. Code is based on that found in the Registrate.py
    file on Github"""
    
    ELASTIX_PATH = os.path.join(elastix_path, "elastix", "elastix.exe")
    if not os.path.exists(ELASTIX_PATH):
        raise IOError('Elastix cannot be found, please set the correct ELASTIX_PATH.')
    
    TRANSFORMIX_PATH = os.path.join(elastix_path, "elastix", "transformix.exe")
    if not os.path.exists(ELASTIX_PATH):
        raise IOError('Elastix cannot be found, please set the correct ELASTIX_PATH.')
    
    result_dir = os.path.join(data_path, "results", fixed_patient)
    if os.path.exists(result_dir) is False:
        os.mkdir(result_dir)
        
    patients = os.listdir(os.path.join(data_path))
    
    for patient in patients:
        if patient[0] == "p" and patient != fixed_patient:
            fixed_image = os.path.join(data_path, fixed_patient, "mr_bffe.mhd")
            fixed_label = os.path.join(data_path, fixed_patient, "prostaat.mhd")
            moving_image = os.path.join(data_path, patient, "mr_bffe.mhd")
            moving_label = os.path.join(data_path, patient, "prostaat.mhd")

            parameter0 = os.path.abspath("Par0001translation.txt")
            parameter1 = os.path.abspath("Par0001bspline64.txt")
            
            output_dir = os.path.join(result_dir,patient)
            if os.path.exists(os.path.join(output_dir)) is False:
                os.mkdir(os.path.abspath(output_dir))
            
            el = elastix.ElastixInterface(elastix_path=ELASTIX_PATH)

            el.register(
                fixed_image=fixed_image,
                moving_image=moving_image,
                parameters=[parameter0, parameter1],
                output_dir=output_dir
            )
            
            t_params = os.path.join(output_dir, "TransformParameters.1.txt")
            
            with open(t_params, 'r') as file:
                lines = file.readlines()

            with open(t_params, 'w') as file:
                for line in lines:
                    if '(FinalBSplineInterpolationOrder 3)' in line:
                        line = line.replace('(FinalBSplineInterpolationOrder 3)', '(FinalBSplineInterpolationOrder 0)')
                    file.write(line)
            
            tr = elastix.TransformixInterface(parameters= t_params, transformix_path=TRANSFORMIX_PATH)

            t_moving_path = os.path.join(output_dir, "mr_bffe")
            t_label_path = os.path.join(output_dir, "prostaat")
                                        
            if os.path.exists(t_moving_path) is False:
                os.mkdir(t_moving_path)
            
            if os.path.exists(t_label_path) is False:
                os.mkdir(t_label_path)
            
            tr.transform_image(moving_image, output_dir= t_moving_path)
            tr.transform_image(moving_label, output_dir=t_label_path)
            
            if show_results is True: 
                fixed_image = sitk.GetArrayFromImage(sitk.ReadImage(fixed_image))
                fixed_label = sitk.GetArrayFromImage(sitk.ReadImage(fixed_label))
                transformed_moving_image = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(t_moving_path, "result.mhd")))
                transformed_moving_label = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(t_label_path, "result.mhd")))
                
                fig, ax = plt.subplots(1, 4, figsize=(20, 5))

                tracker1 = IndexTracker(ax[0], fixed_image)
                tracker2 = IndexTracker(ax[1], fixed_label)
                tracker3 = IndexTracker(ax[2], transformed_moving_image)
                tracker4 = IndexTracker(ax[3], transformed_moving_label)
                fig.canvas.mpl_connect('scroll_event', tracker1.onscroll)
                fig.canvas.mpl_connect('scroll_event', tracker2.onscroll)
                fig.canvas.mpl_connect('scroll_event', tracker3.onscroll)
                fig.canvas.mpl_connect('scroll_event', tracker4.onscroll)
                ax[0].set_title('Fixed image')
                ax[1].set_title('Fixed label')
                ax[2].set_title('Transformed\nmoving image')
                ax[3].set_title('Transformed\nmoving label')
                [x.set_axis_off() for x in ax]
                plt.show()
            
            

def plot_transformed_labels(data_path, fixed_patient, slice_number):
     
    result_dir = os.path.join(data_path, "results", fixed_patient)
    if os.path.exists(result_dir) is False:
        raise IOError('results cannot be found')    
        
    patients = os.listdir(result_dir)
    seg_stack = []
    
    fixed_patient_path = os.path.join(data_path, fixed_patient, "prostaat.mhd")
    image = sitk.ReadImage(str(fixed_patient_path))
    fixed_label = sitk.GetArrayFromImage(image)
    
    for patient in patients:
        if patient[0] == "p":
            seg_path = os.path.join(result_dir, patient, "prostaat", "result.mhd" )
            image = sitk.ReadImage(str(seg_path))
            image_array = sitk.GetArrayFromImage(image)
            seg_stack.append(image_array)
    
    fig, ax = plt.subplots(3, 5, figsize=(30, 20))
    fig.suptitle(f'Transformed Labels and Ground Truth, Slice {slice_number}', fontsize=24)
    
    ax[2,4].imshow(fixed_label[slice_number,:,:])
    ax[2,4].set_title('Ground Truth')
    ax[2,4].set_axis_off()
    
    image_counter = 0
    for j in range (0, 3):
        for i in range(0, 5):
            ax[j,i].imshow(seg_stack[image_counter][slice_number,:,:])
            ax[j,i].set_title(f'Transformed\nmoving label {patients[image_counter]}')
            ax[j,i].set_axis_off()
            image_counter += 1
    
    plt.show()
    
    image_path = os.path.join(data_path, "results", fixed_patient)
    if os.path.exists(image_path) is False:
        os.mkdir(image_path)
        
    fig.savefig(os.path.join(image_path, f"transformed_lables_slice{slice_number}.png"))
    
    return

def plot_labels(data_path, fixed_patient, slice_number):  
        
    patients = os.listdir(data_path)
    seg_stack = []
    
    fixed_patient_path = os.path.join(data_path, fixed_patient, "prostaat.mhd")
    image = sitk.ReadImage(str(fixed_patient_path))
    fixed_label = sitk.GetArrayFromImage(image)
    
    for patient in patients:
        if patient != "results" and patient != fixed_patient:
            seg_path = os.path.join(data_path, patient, "prostaat.mhd")
            image = sitk.ReadImage(str(seg_path))
            image_array = sitk.GetArrayFromImage(image)
            seg_stack.append(image_array)
    
    fig, ax = plt.subplots(3, 5, figsize=(30, 20))
    fig.suptitle(f'Fixed and Moving Labels, Slice {slice_number}', fontsize=24)
    
    ax[2,4].imshow(fixed_label[slice_number,:,:])
    ax[2,4].set_title(f'Fixed Label {fixed_patient}')
    ax[2,4].set_axis_off()
    
    image_counter = 0
    for j in range (0, 3):
        for i in range(0, 5):
            ax[j,i].imshow(seg_stack[image_counter][slice_number,:,:])
            ax[j,i].set_title(f'Moving label {patients[image_counter]}')
            ax[j,i].set_axis_off()
            image_counter += 1
    
    plt.show()
    
    image_path = os.path.join(data_path, "results", fixed_patient)
    if os.path.exists(image_path) is False:
        os.mkdir(image_path)
        
    fig.savefig(os.path.join(image_path, f"fixed_moving_labels_slice{slice_number}"))
    return

def dice_score(fixed_label, moving_label):
    
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    
    for slice in range(fixed_label.shape[0]):
        for y in range(fixed_label.shape[1]):
            for x in range(fixed_label.shape[2]):
                if moving_label[slice, y, x]==1 and fixed_label[slice, y, x]==1:
                    TP += 1
                elif moving_label[slice, y, x]==0 and fixed_label[slice, y, x]==0:
                    TN +=1
                elif moving_label[slice, y, x]==1 and fixed_label[slice, y, x]==0:
                    FP += 1
                elif moving_label[slice, y, x]==0 and fixed_label[slice, y, x]==1:
                    FN +=1

    dice_score = 2*TP / ((TP + FP) + (TP + FN))
    print(dice_score)
    
    return dice_score

def calc_dice_scores(data_path, fixed_patient, staple_label):
    
    result_dir = os.path.join(data_path, "results", fixed_patient)
    if os.path.exists(result_dir) is False:
        raise IOError('results cannot be found')
     
    fixed_patient_path = os.path.join(data_path, fixed_patient, "prostaat.mhd")
    image = sitk.ReadImage(str(fixed_patient_path))
    fixed_label = sitk.GetArrayFromImage(image)
    
    df_scores = pd.DataFrame(columns=['dice score'])
    
    patients = os.listdir(result_dir)
    
    for patient in patients:
        if patient[0] == "p" and patient != fixed_patient:
            seg_path = os.path.join(result_dir, patient, "prostaat", "result.mhd" )
            image = sitk.ReadImage(str(seg_path))
            moving_label = sitk.GetArrayFromImage(image)
            dice = dice_score(fixed_label, moving_label)
            df_scores.loc[patient] = [dice]
    
    staple_dice = dice_score(fixed_label, staple_label)
    df_scores.loc["STAPLE"] = [staple_dice]
    
    df_scores.to_csv(os.path.join(result_dir, 'similarity_measures.csv'))
    
    return df_scores
        
    
if __name__ == "__main__":
    
    elastix_path = os.path.join(r"C:\Users\20182371\Documents\TUe\8DM20_CS_Medical_Imaging")
    data_path = os.path.join(r"C:\Users\20182371\Documents\TUe\8DM20_CS_Medical_Imaging\Data")      
    fixed_patient = "p107"
    slice_number = 50
    
    #registration_transformation(elastix_path, data_path, fixed_patient, True)
    
    staple_label = staple(data_path, fixed_patient)
    
    fixed_patient_path = os.path.join(data_path, fixed_patient, "prostaat.mhd")
    image = sitk.ReadImage(str(fixed_patient_path))
    fixed_label = sitk.GetArrayFromImage(image)
    
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle(f'Fixed and staple, Slice {slice_number}', fontsize=24)
    
    ax[0].imshow(fixed_label[slice_number,:,:])
    ax[0].set_title(f'Fixed label {fixed_patient}')
    ax[0].set_axis_off()
    
    ax[1].imshow(staple_label[slice_number,:,:])
    ax[1].set_title("STAPLE label")
    ax[1].set_axis_off()

    #calc_dice_scores(data_path,fixed_patient, staple_label)
    
    #plot_transformed_labels(data_path, fixed_patient, slice_number)
    #plot_labels(data_path, fixed_patient, slice_number)