# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 16:09:56 2023

@author: 20192024
"""

import os
import pandas as pd
pd.set_option('display.max_columns', None)
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
import SimpleITK as sitk
from scipy.spatial.distance import directed_hausdorff

# Define paths and load data

experiment1 = 'Translation'
experiment2 = 'TransAffine'

#still needs to compute from here
experiment3 = 'TransRigid'
experiment4 = 'TransBSpline'
experiment5 = 'TransRigidBSpline'
experiment6 = 'TransAffineBSpline'

threshold = 0.5
threshold_staple = 0.95

seed = 1
path = os.path.join('D:\\CapitaSelecta\\', 'results_registration_seed'+str(seed))

gt_path = 'C:\\Users\\20192024\\OneDrive - TU Eindhoven\Documents\\Y4\\Q3\\Capita Selecta in Medical Image Analysis\\Project\\Nieuw\\TrainingData'

    
#exp_list = [experiment1, experiment2, experiment3, experiment4, experiment5, experiment6]

exp_list = [experiment3, experiment4, experiment5, experiment6]


directories = os.listdir(os.path.join(path, experiment3))
check = 'p'
fixed_patients_list = [idx for idx in directories if idx[0].lower() == check.lower()]

scores = {}

for experiment in exp_list:
    for fixed_patient in fixed_patients_list:
        
        ground_truth = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(gt_path, fixed_patient, 'prostaat.mhd'))).astype(np.int16)
        
        seg_stack = []
        
        fixed_results_path = os.path.join(path,experiment,fixed_patient)
        list_atlas = os.listdir(fixed_results_path)
        for atlas in list_atlas:
            
            segmentation = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(fixed_results_path, atlas, "result.mhd")))
            segmentation = np.nan_to_num(segmentation)
            segmentation = (segmentation > threshold).astype(np.int16)
            
            hausdorf = []
            for slice in range(segmentation.shape[0]):
                hausdorf.append(directed_hausdorff(segmentation[slice, :, :], ground_truth[slice, :, :])[0])
            hausdorf = np.array(hausdorf)
            mean_hausdorf = hausdorf.mean()
            std_hausdorf = hausdorf.std()
            scores[fixed_patient, atlas] = [mean_hausdorf, std_hausdorf]
            
            
            
            seg_stack.append(sitk.GetImageFromArray(segmentation))
            
        staple = sitk.STAPLE(seg_stack, 1.0) 
        staple = sitk.GetArrayFromImage(staple)
        staple = (staple > threshold_staple).astype(np.int16)
        
        hausdorf = []
        for slice in range(staple.shape[0]):
            hausdorf.append(directed_hausdorff(staple[slice, :, :], ground_truth[slice, :, :])[0])
        hausdorf = np.array(hausdorf)
        mean_hausdorf_stap = hausdorf.mean()
        std_hausdorf_stap = hausdorf.std()
        scores[fixed_patient, "STAPLE"] = [mean_hausdorf, std_hausdorf]
        
    
    csv = pd.DataFrame.from_dict(scores)
    csv = csv.transpose()
    csv.columns=["Hausdorff distance mean", "Hausdorff distance std"]
    csv.to_csv(os.path.join(path,experiment,"scores_new.csv"))
        
        
        


# def calculate_scores(self):
#     self.segmentations = {}
#     self.ground_truths = {}
#     self.scores = {}
#     print("Running for parameters", self.config.parameters)
#     for valid_i in tqdm(self.config.valid_indx):
#         ground_truth = sitk.ReadImage(os.path.join(self.config.folder_preprocessed, self.config.patients[valid_i], "prostaat.mhd"))
#         ground_truth = sitk.GetArrayFromImage(ground_truth).astype(np.int16)
#         self.ground_truths[self.config.patients[valid_i]] = ground_truth
#         seg_stack = []
#         for train_i in self.config.train_indx:
#             segmentation = sitk.ReadImage(os.path.join(self.config.folder_results, self.config.now, self.config.patients[valid_i], self.config.patients[train_i], "result.mhd"))
#             segmentation = sitk.GetArrayFromImage(segmentation)
#             segmentation = np.nan_to_num(segmentation)
#             segmentation = (segmentation > self.threshold).astype(np.int16)
            
#             dice = dice_score(segmentation, ground_truth)

#             hausdorf = []
#             for slice in range(segmentation.shape[0]):
#                 hausdorf.append(directed_hausdorff(segmentation[slice, :, :], ground_truth[slice, :, :]))
#             hausdorf = np.array(hausdorf)
#             mean_hausdorf = hausdorf.mean()
#             std_hausdorf = hausdorf.std()
#             self.scores[self.config.patients[valid_i], self.config.patients[train_i]] = [dice, mean_hausdorf, std_hausdorf]

#             self.segmentations[self.config.patients[valid_i], self.config.patients[train_i]] = segmentation                
#             seg_stack.append(sitk.GetImageFromArray(segmentation))
       
#         staple = sitk.STAPLE(seg_stack, 1.0) 
#         staple = sitk.GetArrayFromImage(staple)
#         staple = (staple > self.threshold_staple).astype(np.int16)
        
#         sitk.WriteImage(sitk.GetImageFromArray(staple), os.path.join(self.config.folder_results, self.config.now, self.config.patients[valid_i], 'staple.mhd'))
        
        
        
#         dice = dice_score(staple, ground_truth)
        
#         hausdorf = []
#         for slice in range(staple.shape[0]):
#             hausdorf.append(directed_hausdorff(staple[slice, :, :], ground_truth[slice, :, :]))
#         hausdorf = np.array(hausdorf)
#         mean_hausdorf = hausdorf.mean()
#         std_hausdorf = hausdorf.std()
#         self.scores[self.config.patients[valid_i], "STAPLE"] = [dice, mean_hausdorf, std_hausdorf]

#         self.segmentations[self.config.patients[valid_i], "STAPLE"] = staple

#     csv = pd.DataFrame.from_dict(self.scores)
#     csv = csv.transpose()
#     csv.columns=["Dice Score", "Hausdorff distance mean", "Hausdorff distance std"]
#     csv.to_csv(os.path.join(self.config.folder_results, self.config.now, "scores.csv"))