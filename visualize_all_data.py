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

csv_path1 = os.path.join(path, experiment1, 'scores.csv')
csv_path2 = os.path.join(path, experiment2, 'scores.csv')
csv_path3 = os.path.join(path, experiment3, 'scores.csv')
csv_path4 = os.path.join(path, experiment4,'scores.csv')
csv_path5 = os.path.join(path, experiment5, 'scores.csv')
csv_path6 = os.path.join(path, experiment6, 'scores.csv')


df1 = pd.read_csv(csv_path1)
df2 = pd.read_csv(csv_path2)
df3 = pd.read_csv(csv_path3)
df4 = pd.read_csv(csv_path4)
df5 = pd.read_csv(csv_path5)
df6 = pd.read_csv(csv_path6)

df1 = df1.rename(columns={'Unnamed: 0': 'fixed patient', 'Unnamed: 1':'moving patient'})
df2 = df2.rename(columns={'Unnamed: 0': 'fixed patient', 'Unnamed: 1':'moving patient'})
df3 = df3.rename(columns={'Unnamed: 0': 'fixed patient', 'Unnamed: 1':'moving patient'})
df4 = df4.rename(columns={'Unnamed: 0': 'fixed patient', 'Unnamed: 1':'moving patient'})
df5 = df5.rename(columns={'Unnamed: 0': 'fixed patient', 'Unnamed: 1':'moving patient'})
df6 = df6.rename(columns={'Unnamed: 0': 'fixed patient', 'Unnamed: 1':'moving patient'})


# Find STAPLE rows and allocate to separate dataframe
df_STAPLE1 = df1.loc[df1['moving patient'] == 'STAPLE']
df_STAPLE2 = df2.loc[df2['moving patient'] == 'STAPLE']
df_STAPLE3 = df3.loc[df3['moving patient'] == 'STAPLE']
df_STAPLE4 = df4.loc[df4['moving patient'] == 'STAPLE']
df_STAPLE5 = df5.loc[df5['moving patient'] == 'STAPLE']
df_STAPLE6 = df6.loc[df6['moving patient'] == 'STAPLE']

STAPLE_list = [df_STAPLE1, df_STAPLE2, df_STAPLE3, df_STAPLE4, df_STAPLE5, df_STAPLE6]


df1 = df1.loc[df1['moving patient'] != 'STAPLE']
df2 = df2.loc[df2['moving patient'] != 'STAPLE']
df3 = df3.loc[df3['moving patient'] != 'STAPLE']
df4 = df4.loc[df4['moving patient'] != 'STAPLE']
df5 = df5.loc[df5['moving patient'] != 'STAPLE']
df6 = df6.loc[df6['moving patient'] != 'STAPLE']

df_dice = df1[['Dice Score']].copy()
df_dice[experiment2] = df2['Dice Score']
df_dice[experiment3] = df3['Dice Score']
df_dice[experiment4] = df4['Dice Score']
df_dice[experiment5] = df5['Dice Score']
df_dice[experiment6] = df6['Dice Score']
df_dice.rename(columns={ df_dice.columns[0]: experiment1 }, inplace = True)

df_haus = df1[['Hausdorff distance mean']].copy()
df_haus[experiment2] = df2['Hausdorff distance mean']
df_haus[experiment3] = df3['Hausdorff distance mean']
df_haus[experiment4] = df4['Hausdorff distance mean']
df_haus[experiment5] = df5['Hausdorff distance mean']
df_haus[experiment6] = df6['Hausdorff distance mean']
df_haus.rename(columns={ df_haus.columns[0]: experiment1 }, inplace = True)

# Plotting everything
boxprops = dict(linestyle='-', linewidth=5)
whiskerprops = dict(linestyle='-', linewidth=5)
capprops = dict(linestyle='-', linewidth=5)
medianprops = dict(linestyle='-', linewidth=5)

# Dice score
fig, ax = plt.subplots(1,1)

df_dice.boxplot(ax = ax, boxprops=boxprops, whiskerprops=whiskerprops, capprops=capprops, medianprops=medianprops)
fig.set_size_inches(26,18)
title = 'Dice scores for different transformation combinations - split 2'
fig.suptitle(title, fontsize=45, fontweight='bold')
fig.tight_layout(pad=10.0)

for i in range(6):
    df_stap_exp = STAPLE_list[i]
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


# Hausdorff

fig, ax = plt.subplots(1,1)

df_haus.boxplot(ax = ax, boxprops=boxprops, whiskerprops=whiskerprops, capprops=capprops, medianprops=medianprops)
fig.set_size_inches(26,18)
title = 'Hausdorff distances for different transformation combinations - split 2'
fig.suptitle(title, fontsize=45, fontweight='bold')
fig.tight_layout(pad=10.0)

for i in range(6):
    df_stap_exp = STAPLE_list[i]
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

dfs = [df1, df2, df3, df4, df5, df6]
p_dice = []
p_haus =[]

for df in dfs:
    fix1 = df.loc[df['fixed patient'] == 'p109']
    fix2 = df.loc[df['fixed patient'] == 'p119']
    fix3 = df.loc[df['fixed patient'] == 'p117']
    
    p_dice.append({'fix1':stats.shapiro(fix1['Dice Score'])[1], 'fix2':stats.shapiro(fix2['Dice Score'])[1], 'fix3':stats.shapiro(fix3['Dice Score'])[1], 'all': stats.shapiro(df['Dice Score'])[1]})
    p_haus.append({'fix1':stats.shapiro(fix1['Hausdorff distance mean'])[1], 'fix2':stats.shapiro(fix2['Hausdorff distance mean'])[1], 'fix3':stats.shapiro(fix3['Hausdorff distance mean'])[1], 'all': stats.shapiro(df['Hausdorff distance mean'])[1]})

# Create heatmap of p-values
p_values = np.zeros((6,6))

column_names = []
for col in df_dice.columns:
    column_names.append(col)

for i in range(len(column_names)):
    for j in range(len(column_names)):
        p_values[i,j] = stats.wilcoxon(df_dice[column_names[i]], df_dice[column_names[j]])[1]
        


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

# =============================================================================
# Code to visualize and compare metric results from different patient splits
# =============================================================================






# =============================================================================
# Code to visualize and compare metric results from different grid spacings
# =============================================================================


