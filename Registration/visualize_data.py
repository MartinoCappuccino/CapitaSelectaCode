# -*- coding: utf-8 -*-
"""
Created on Sun Mar  5 16:04:34 2023

@author: 20192024
"""

import os
import pandas as pd
pd.set_option('display.max_columns', None)
import numpy as np
import matplotlib.pyplot as plt

SMALL_SIZE = 8
MEDIUM_SIZE = 16
BIGGER_SIZE = 25

experiment = 'TranslationRigid'

# In case the data is on USB
path_to_experiment = os.path.join('D:\\CapitaSelecta\\results_registration', experiment)
csv_path = os.path.join(path_to_experiment, 'scores.csv')

# In case the data is on laptop
# path_to_experiment = os.path.join('C:\\Users\\20192024\\OneDrive - TU Eindhoven\\Documents\\Y4\\Q3\\Capita Selecta in Medical Image Analysis\\Project\\Nieuw\\results', experiment)
# csv_path = os.path.join(path_to_experiment, 'scores.csv')


df = pd.read_csv(csv_path)
df = df.rename(columns={'Unnamed: 0': 'fixed patient', 'Unnamed: 1':'moving patient'})

# Find STAPLE rows and allocate to separate dataframe
df_STAPLE = df.loc[df['moving patient'] == 'STAPLE']
df_raw = df.loc[df['moving patient'] != 'STAPLE']

# =============================================================================
# # Plot per fixed patient
# =============================================================================

fig, ax = plt.subplots(1,2)
df_raw.boxplot(ax = ax[0], column=['Dice Score'], by=['fixed patient'])
#, medianprops={"linewidth": 8, "solid_capstyle": "butt"}
df_raw.boxplot(ax = ax[1], column=['Hausdorff distance mean'], by=['fixed patient'])
fig.set_size_inches(9,8)
title = experiment + ' per fixed patient'
fig.suptitle(title, fontsize=22, fontweight='bold')
fig.tight_layout(pad=5.0)


for i in range(len(df_STAPLE)):
    y_dice = df_STAPLE['Dice Score'].iloc[i]
    y_haus = df_STAPLE['Hausdorff distance mean'].iloc[i]
    ax[0].plot(i+1, y_dice, color='r', marker='*', markersize=9.5) 
    ax[1].plot(i+1, y_haus, color='r',  marker='*', markersize=9.5) 

# set xaxis and yaxis fonts
for i in range(2):
    for item in ([ax[i].xaxis.label, ax[i].yaxis.label] +
             ax[i].get_xticklabels() + ax[i].get_yticklabels()):
        item.set_fontsize(MEDIUM_SIZE)

ax[0].set_xlabel('Fixed patient')
ax[1].set_xlabel('Fixed patient')
ax[0].title.set_fontsize(20)
ax[1].title.set_fontsize(20)

save_path = os.path.join(path_to_experiment,'boxplot_per_fixedpatient.png')
plt.savefig(save_path, dpi=200)

# =============================================================================
# Plot with average scores of all patients
# =============================================================================

fig, ax = plt.subplots(1,2)
df_raw.boxplot(ax = ax[0], column=['Dice Score'])
df_raw.boxplot(ax = ax[1], column=['Hausdorff distance mean'])
fig.set_size_inches(9,8)
title = experiment + ' over all patients'
fig.suptitle(title, fontsize=22, fontweight='bold')
fig.tight_layout(pad=7.0)

# Plot median STAPLE scores for Dice and Hausdorff separately
# Check with the rest: plot median or mean STAPLE?

ax[0].plot(1, df_STAPLE['Dice Score'].median(), color='r', marker='*', markersize=9.5) 
ax[1].plot(1, df_STAPLE['Hausdorff distance mean'].median(), color='r',  marker='*', markersize=9.5) 


for i in range(2):
    for item in ([ax[i].xaxis.label, ax[i].yaxis.label] +
             ax[i].get_xticklabels() + ax[i].get_yticklabels()):
        item.set_fontsize(MEDIUM_SIZE)


ax[0].title.set_fontsize(20)
ax[1].title.set_fontsize(20)
ax[0].title.set_text('Dice Score')
ax[1].title.set_text('Hausdorff distance mean')
ax[0].set_xticks([])
ax[1].set_xticks([])


save_path = os.path.join(path_to_experiment,'boxplot_total.png')
plt.savefig(save_path, dpi=200)


