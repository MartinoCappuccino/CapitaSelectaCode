# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 13:10:53 2023

@author: 20192024
"""
import os
import pandas as pd
pd.set_option('display.max_columns', None)
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
# =============================================================================
# Code to visualize and compare metric results from different transformations
# =============================================================================

# Define paths

experiment1 = 'Translation_seed1'
experiment2 = 'TransAffine_seed1'
experiment3 = 'TransRigid_seed1'
experiment4 = 'TransBSpline_seed1'
experiment5 = 'TransRigidBSpline_seed1'
experiment6 = 'TransAffineBSpline_seed1'

path = 'D:\\CapitaSelecta\\results_registration_seed1'

exp_list = [experiment1, experiment2, experiment3, experiment4, experiment5, experiment6]
df_dict = {}
df_staple_dict = {}


for experiment in exp_list:
    df_dict[experiment] = pd.read_csv(os.path.join(path, experiment, 'scores.csv'))
    df_dict[experiment] = df_dict[experiment].rename(columns={'Unnamed: 0': 'fixed patient', 'Unnamed: 1':'moving patient'})
    
    # Find STAPLE rows and allocate to separate dataframe
    df_staple_dict[experiment] = df_dict[experiment].loc[df_dict[experiment]['moving patient'] == 'STAPLE']
    
    # We remove STAPLE rows from the first (original) dataframe to obtain only the 'raw' results
    df_dict[experiment] = df_dict[experiment].loc[df_dict[experiment]['moving patient'] != 'STAPLE']
    
# Only extracting dice scores to make separate dataframe    
dict_variable_dice = {exp:df_dict[exp]['Dice Score'] for exp in exp_list}
df_dice = pd.DataFrame(dict_variable_dice)

dict_variable_haus = {exp:df_dict[exp]['Hausdorff distance mean'] for exp in exp_list}
df_haus = pd.DataFrame(dict_variable_haus)


# Plotting everything
boxprops = dict(linestyle='-', linewidth=5)
whiskerprops = dict(linestyle='-', linewidth=5)
capprops = dict(linestyle='-', linewidth=5)
medianprops = dict(linestyle='-', linewidth=5)

# DICE
fig, ax = plt.subplots(1,1)

df_dice.boxplot(ax = ax, boxprops=boxprops, whiskerprops=whiskerprops, capprops=capprops, medianprops=medianprops)
fig.set_size_inches(26,18)
title = 'Dice scores for different transformation combinations - split 2'
fig.suptitle(title, fontsize=45, fontweight='bold')
fig.tight_layout(pad=10.0)

for i in range(len(exp_list)):
    df_stap_exp = df_staple_dict[exp_list[i]]
    median = df_stap_exp['Dice Score'].median()
    ax.plot(i+1, median, color='r', marker='*', markersize=24) 

xticks = ['T', 'T+A', 'T+R', 'T+BS', 'T+R+BS', 'T+A+BS']
ax.set_xticklabels(xticks)

labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontweight('bold') for label in labels]

ax.tick_params(axis='x', labelrotation=0, labelsize=35)
ax.tick_params(axis='y', labelrotation=0, labelsize=40)

ax.set_ylabel('Dice score', fontsize=40)
ax.set_xlabel('Transformation', fontsize=40)
ax.yaxis.labelpad = 20
ax.xaxis.labelpad = 20

save_path = os.path.join(path,'dice_score_different_trans_split2.png')
plt.savefig(save_path, dpi=200, bbox_inches="tight")


# HAUSDORFF

fig, ax = plt.subplots(1,1)

df_haus.boxplot(ax = ax, boxprops=boxprops, whiskerprops=whiskerprops, capprops=capprops, medianprops=medianprops)
fig.set_size_inches(26,18)
title = 'Hausdorff distances for different transformation combinations - split 2'
fig.suptitle(title, fontsize=45, fontweight='bold')
fig.tight_layout(pad=10.0)

for i in range(len(exp_list)):
    df_stap_exp = df_staple_dict[exp_list[i]]
    median = df_stap_exp['Hausdorff distance mean'].median()
    ax.plot(i+1, median, color='r', marker='*', markersize=24) 


xticks = ['T', 'T+A', 'T+R', 'T+BS', 'T+R+BS', 'T+A+BS']
ax.set_xticklabels(xticks)

labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontweight('bold') for label in labels]

ax.tick_params(axis='x', labelrotation=0, labelsize=35)
ax.tick_params(axis='y', labelrotation=0, labelsize=40)

ax.set_ylabel('Hausdorff distance', fontsize=45)
ax.set_xlabel('Transformation', fontsize=45)
ax.yaxis.labelpad = 20
ax.xaxis.labelpad = 20

save_path = os.path.join(path,'haus_different_trans_split2.png')
plt.savefig(save_path, dpi=200, bbox_inches="tight")

# =============================================================================
# Statistical testing
# =============================================================================

# Shapiro Wilk per fixed patient
# Dice score

p_dice = []
p_haus =[]

for experiment in exp_list:
    fix1 = df_dict[experiment].loc[df_dict[experiment]['fixed patient'] == 'p109']
    fix2 = df_dict[experiment].loc[df_dict[experiment]['fixed patient'] == 'p119']
    fix3 = df_dict[experiment].loc[df_dict[experiment]['fixed patient'] == 'p117']
    
    p_dice.append({'fix1':stats.shapiro(fix1['Dice Score'])[1], 'fix2':stats.shapiro(fix2['Dice Score'])[1], 'fix3':stats.shapiro(fix3['Dice Score'])[1], 'all': stats.shapiro(df_dict[experiment]['Dice Score'])[1]})
    p_haus.append({'fix1':stats.shapiro(fix1['Hausdorff distance mean'])[1], 'fix2':stats.shapiro(fix2['Hausdorff distance mean'])[1], 'fix3':stats.shapiro(fix3['Hausdorff distance mean'])[1], 'all': stats.shapiro(df_dict[experiment]['Hausdorff distance mean'])[1]})

# Create heatmap
p_values = np.zeros((6,6))


for i in range(len(exp_list)):
    for j in range(len(exp_list)):
        if i == j:
            p_values[i,j] = None
        else:
            p_values[i,j] = stats.wilcoxon(df_dict[exp_list[i]]['Dice Score'], df_dict[exp_list[j]]['Dice Score'])[1]
        

ax = sns.heatmap(p_values, linewidth=0.5)
ax.set_title('Heatmap of p-values - Dice scores - split 2')
ax.set_xticklabels(xticks)
ax.set_yticklabels(xticks)
ax.tick_params(axis='x', labelrotation=0, labelsize=8.5)
ax.tick_params(axis='y', labelrotation=0, labelsize=8.5)

plt.show()
save_path = os.path.join(path,'dice_p_values_Wilcox_split2.png')

fig = ax.get_figure()
fig.savefig(save_path, dpi=200)

