# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 15:57:40 2023

@author: 20192757
"""
import matplotlib.pyplot as plt
import statistics as st
from scipy import stats
import seaborn as sns
import numpy as np

score0 = []
score = "HD"
with open("{x}_0.txt".format(x=score), "r") as f:
  for line in f:
      x = (line.strip())
      score0.append(float(x))

score4 = [] 
with open("{x}_4.txt".format(x=score), "r") as f:
  for line in f:
      x = (line.strip())
      score4.append(float(x))
    
score8 = [] 
with open("{x}_8.txt".format(x=score), "r") as f:
  for line in f:
      x = line.strip()
      score8.append(float(x))

score12 = [] 
with open("{x}_12.txt".format(x=score), "r") as f:
  for line in f:
      x = line.strip()
      score12.append(float(x))

score16 = [] 
with open("{x}_16.txt".format(x=score), "r") as f:
  for line in f:
      x = line.strip()
      score16.append(float(x))

score20 = [] 
with open("{x}_20.txt".format(x=score), "r") as f:
  for line in f:
      x = line.strip()
      score20.append(float(x))

# score24 = [] 
# with open("{x}_24.txt".format(x=score), "r") as f:
#   for line in f:
#       x = line.strip()
#       score24.append(float(x))

score28 = [] 
with open("{x}_28.txt".format(x=score), "r") as f:
  for line in f:
      x = line.strip()
      score28.append(float(x))
      
score32 = [] 
with open("{x}_32.txt".format(x=score), "r") as f:
  for line in f:
      x = line.strip()
      score32.append(float(x))


score24 = [0]*258

scores = [score0, score4, score8, score12, score16, score20, score24, score28, score32]
p_values = np.zeros((9,9))
column_names = ['0', '4', '8','12','16','20','24','28','32']

for i in range(len(column_names)):
      for j in range(len(column_names)):
          if i != j: 
              p_values[i,j] = stats.wilcoxon(scores[i],scores[j])[1]
          else: 
              p_values[i,j] = 1

#P_value = stats.wilcoxon(score0,score0)
#print(P_value)
ax = sns.heatmap(p_values, linewidth=0.5)
ax.set_title('Heatmap of p-values per number of fake images - HD scores')
ax.set_xticklabels(column_names)
ax.set_yticklabels(column_names)
ax.tick_params(axis='x', labelrotation=0, labelsize=8.5)
ax.tick_params(axis='y', labelrotation=0, labelsize=8.5)

plt.show()
fig = ax.get_figure()
fig.savefig('HD_HM')