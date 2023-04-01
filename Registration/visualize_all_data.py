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

experiment1 = 'Translation'
experiment2 = 'TransAffine'
experiment3 = 'TransRigid'
experiment4 = 'TransBSpline'
experiment5 = 'TransRigidBSpline'
experiment6 = 'TransAffineBSpline'

seed = 0
path = os.path.join('D:\\CapitaSelecta\\', 'results_registration_seed'+str(seed))
directories = os.listdir(os.path.join(path, experiment3))
check = 'p'
fixed_patients = [idx for idx in directories if idx[0].lower() == check.lower()]


exp_list = [experiment1, experiment2, experiment3, experiment4, experiment5, experiment6]
df_dict = {}
df_staple_dict = {}


for experiment in exp_list:
    df_dict[experiment] = pd.read_csv(os.path.join(path, experiment, 'scores_new.csv'))
    df_dict[experiment] = df_dict[experiment].rename(columns={'Unnamed: 0': 'fixed patient', 'Unnamed: 1':'moving patient'})
    
    # Find STAPLE rows and allocate to separate dataframe
    df_staple_dict[experiment] = df_dict[experiment].loc[df_dict[experiment]['moving patient'] == 'STAPLE']
    
    # We remove STAPLE rows from the first (original) dataframe to obtain only the 'raw' results
    df_dict[experiment] = df_dict[experiment].loc[df_dict[experiment]['moving patient'] != 'STAPLE']
    
# Only extracting dice scores to make separate dataframe    
# dict_variable_dice = {exp:df_dict[exp]['Dice Score'] for exp in exp_list}
# df_dice = pd.DataFrame(dict_variable_dice)

dict_variable_haus = {exp:df_dict[exp]['Hausdorff distance mean'] for exp in exp_list}
df_haus = pd.DataFrame(dict_variable_haus)


# Plotting everything
boxprops = dict(linestyle='-', linewidth=5)
whiskerprops = dict(linestyle='-', linewidth=5)
capprops = dict(linestyle='-', linewidth=5)
medianprops = dict(linestyle='-', linewidth=5)

# DICE
# fig, ax = plt.subplots(1,1)

# df_dice.boxplot(ax = ax, boxprops=boxprops, whiskerprops=whiskerprops, capprops=capprops, medianprops=medianprops)
# fig.set_size_inches(26,18)
# title = 'Dice scores for different transformation combinations - split 1'
# fig.suptitle(title, fontsize=45, fontweight='bold')
# fig.tight_layout(pad=10.0)

# for i in range(len(exp_list)):
#     df_stap_exp = df_staple_dict[exp_list[i]]
#     median = df_stap_exp['Dice Score'].median()
#     ax.plot(i+1, median, color='r', marker='*', markersize=24) 

# xticks = ['T', 'T+A', 'T+R', 'T+BS', 'T+R+BS', 'T+A+BS']
# ax.set_xticklabels(xticks)

# labels = ax.get_xticklabels() + ax.get_yticklabels()
# [label.set_fontweight('bold') for label in labels]

# ax.tick_params(axis='x', labelrotation=0, labelsize=35)
# ax.tick_params(axis='y', labelrotation=0, labelsize=40)

# ax.set_ylabel('Dice score', fontsize=40)
# ax.set_xlabel('Transformation', fontsize=40)
# ax.yaxis.labelpad = 20
# ax.xaxis.labelpad = 20

# save_path = os.path.join(path,'dice_score_different_trans_split1.png')
# plt.savefig(save_path, dpi=200, bbox_inches="tight")


# HAUSDORFF

fig, ax = plt.subplots(1,1)

df_haus.boxplot(ax = ax, boxprops=boxprops, whiskerprops=whiskerprops, capprops=capprops, medianprops=medianprops)
fig.set_size_inches(26,18)
title = 'Mean Hausdorff distances for different transformation combinations - split ' + str(seed+1)
fig.suptitle(title, fontsize=45, fontweight='bold')
plt.ylim(0,8)
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

save_path = os.path.join(path,'haus_mean_different_trans_split' + str(seed+1)+'_new.png')
plt.savefig(save_path, dpi=200, bbox_inches="tight")

# =============================================================================
# Statistical testing
# =============================================================================

# Shapiro Wilk per fixed patient
# Dice score

#p_dice = []
p_haus =[]

for experiment in exp_list:
    fix1 = df_dict[experiment].loc[df_dict[experiment]['fixed patient'] == fixed_patients[0]]
    fix2 = df_dict[experiment].loc[df_dict[experiment]['fixed patient'] == fixed_patients[1]]
    fix3 = df_dict[experiment].loc[df_dict[experiment]['fixed patient'] == fixed_patients[2]]
    
    #p_dice.append({'fix1':stats.shapiro(fix1['Dice Score'])[1], 'fix2':stats.shapiro(fix2['Dice Score'])[1], 'fix3':stats.shapiro(fix3['Dice Score'])[1], 'all': stats.shapiro(df_dict[experiment]['Dice Score'])[1]})
    p_haus.append({'fix1':stats.shapiro(fix1['Hausdorff distance mean'])[1], 'fix2':stats.shapiro(fix2['Hausdorff distance mean'])[1], 'fix3':stats.shapiro(fix3['Hausdorff distance mean'])[1], 'all': stats.shapiro(df_dict[experiment]['Hausdorff distance mean'])[1]})

# Create heatmap
p_values = np.zeros((6,6))


for i in range(len(exp_list)):
    for j in range(len(exp_list)):
        if i == j:
            p_values[i,j] = None
        else:
            p_values[i,j] = stats.ttest_rel(df_dict[exp_list[i]]['Hausdorff distance mean'], df_dict[exp_list[j]]['Hausdorff distance mean'])[1]
        
fig, ax = plt.subplots(1,1)
ax = sns.heatmap(p_values, linewidth=0.5)
ax.set_title('Heatmap of p-values - Hausdorff distances - split ' + str(seed+1))
ax.set_xticklabels(xticks)
ax.set_yticklabels(xticks)
ax.tick_params(axis='x', labelrotation=0, labelsize=8.5)
ax.tick_params(axis='y', labelrotation=0, labelsize=8.5)

plt.show()
save_path = os.path.join(path,'haus_p_values_Wilcox_split'+str(seed+1)+'_new.png')

fig = ax.get_figure()
fig.savefig(save_path, dpi=200)


# =============================================================================
# Even more visualisation! :D
# =============================================================================


df_haus.insert(0, 'Fixed patient', df_dict[experiment1]['fixed patient'])
df_long = pd.melt(df_haus, "Fixed patient", var_name="Experiment", value_name="Hausdorff")
a4_dims = (18, 8.27)
fig, ax = plt.subplots(figsize=a4_dims)
plt.ylim(0,8)

sns.factorplot("Experiment", hue="Fixed patient", y="Hausdorff", data=df_long, kind="box", ax=ax, width=0.5)
fig.suptitle('Hausdorff distance scores for Registration for split ' + str(seed+1), fontsize=20)
ax.tick_params(axis='x', labelrotation=0, labelsize=15)
ax.tick_params(axis='y', labelrotation=0, labelsize=15)
ax.set_ylabel('Hausdorff', fontsize=15, fontweight='bold')
ax.set_xlabel('Experiment', fontsize=15, fontweight='bold')
plt.setp(ax.get_legend().get_texts(), fontsize='14') # for legend text
plt.setp(ax.get_legend().get_title(), fontsize='14') # for legend title



# also plotting STAPLE

dict_variable_staple = {exp:df_staple_dict[exp]['Hausdorff distance mean'] for exp in exp_list}
df_STAPLE = pd.DataFrame(dict_variable_staple)
df_STAPLE.insert(0, 'Fixed patient', df_staple_dict[experiment1]['fixed patient'])
df_long_staple = pd.melt(df_STAPLE, "Fixed patient", var_name="Experiment", value_name="Hausdorff")


#for i in range(len(exp_list)): 
#    ax.annotate(df_staple_dict[exp_list[i]]['Hausdorff distance mean'], xy=(i, df_staple_dict[exp_list[i]]['Hausdorff distance mean'])) 

def add_median_labels(ax, fmt='.1f'):
    lines = ax.get_lines()
    boxes = [c for c in ax.get_children() if type(c).__name__ == 'PathPatch']
    lines_per_box = int(len(lines) / len(boxes))
    for median in lines[4:len(lines):lines_per_box]:
        x, y = (data.mean() for data in median.get_data())
        # choose value depending on horizontal or vertical plot orientation
        value = x if (median.get_xdata()[1] - median.get_xdata()[0]) == 0 else y
        text = ax.text(x, y, f'{value:{fmt}}', ha='center', va='center',
                       fontweight='bold', color='white')
        # create median-colored border around white text for contrast
        text.set_path_effects([
            path_effects.Stroke(linewidth=3, foreground=median.get_color()),
            path_effects.Normal(),
        ])








save_path = os.path.join(path,'Overall_haus_new.png')
fig.savefig(save_path, dpi=200)







