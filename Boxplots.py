# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 10:28:34 2023

@author: 20192757
"""
import matplotlib.pyplot as plt
import statistics as st
from scipy import stats

score0 = []
score = "DICE"
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




# score0 = list(filter(lambda num: num != 0, score0))
# score4 = list(filter(lambda num: num != 0, score4))
# score8 = list(filter(lambda num: num != 0, score8))
# score12 = list(filter(lambda num: num != 0, score8))
# #score16 = list(filter(lambda num: num != 0, score8))
# score20 = list(filter(lambda num: num != 0, score8))
# #score24 = list(filter(lambda num: num != 0, score8))
# score28 = list(filter(lambda num: num != 0, score8))
# score32 = list(filter(lambda num: num != 0, score8))
    
# = [0,0,0]
score24 = [0,0,0]

boxprops = dict(linestyle='-', linewidth=5, color='steelblue')
whiskerprops = dict(linestyle='-', linewidth=5, color='steelblue')
capprops = dict(linestyle='-', linewidth=5, color='black')
medianprops = dict(linestyle='-', linewidth=5, color = 'limegreen')

# DICE
fig, ax = plt.subplots(1,1)
fig.set_size_inches(26,18)
fig.tight_layout(pad=10.0)

#plt.boxplot(ax = ax, boxprops=boxprops, whiskerprops=whiskerprops, capprops=capprops, medianprops=medianprops)
plt.boxplot(score0, positions = [1], widths=(0.6), boxprops=boxprops, whiskerprops=whiskerprops, capprops=capprops, medianprops=medianprops)
plt.boxplot(score4, positions = [2], widths=(0.6),boxprops=boxprops, whiskerprops=whiskerprops, capprops=capprops, medianprops=medianprops)
plt.boxplot(score8, positions = [3], widths=(0.6),boxprops=boxprops, whiskerprops=whiskerprops, capprops=capprops, medianprops=medianprops)
plt.boxplot(score12, positions = [4], widths=(0.6),boxprops=boxprops, whiskerprops=whiskerprops, capprops=capprops, medianprops=medianprops)
plt.boxplot(score16, positions = [5], widths=(0.6),boxprops=boxprops, whiskerprops=whiskerprops, capprops=capprops, medianprops=medianprops)
plt.boxplot(score20, positions = [6], widths=(0.6),boxprops=boxprops, whiskerprops=whiskerprops, capprops=capprops, medianprops=medianprops)
plt.boxplot(score24, positions = [7], widths=(0.6),boxprops=boxprops, whiskerprops=whiskerprops, capprops=capprops, medianprops=medianprops)
plt.boxplot(score28, positions = [8], widths=(0.6),boxprops=boxprops, whiskerprops=whiskerprops, capprops=capprops, medianprops=medianprops)
plt.boxplot(score32, positions = [9], widths=(0.6),boxprops=boxprops, whiskerprops=whiskerprops, capprops=capprops, medianprops=medianprops)

title = 'DSC for different number of fake images'
fig.suptitle(title, fontsize=45, fontweight='bold')

plt.grid()

xticks = ['0', '4', '8','12','16','20','24','28','32']
ax.set_xticklabels(xticks)

labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontweight('bold') for label in labels]

ax.tick_params(axis='x', labelrotation=0, labelsize=35)
ax.tick_params(axis='y', labelrotation=0, labelsize=40)

ax.set_ylabel('DSC', fontsize=40)
ax.set_xlabel('Number of fake images', fontsize=40)
ax.yaxis.labelpad = 20
ax.xaxis.labelpad = 20

plt.savefig('Dice_ML.png')
   
median0 = st.median(score0)
median4 = st.median(score4)
median8 = st.median(score8)
median12 = st.median(score12)
median16 = st.median(score16)
median20 = st.median(score20)
median24 = st.median(score24)
median28 = st.median(score28)
median32 = st.median(score32)
median = [median0, median4, median8, median12, median16, median20, median24, median28, median32]
print(median)

std0 = st.stdev(score0)
std4 = st.stdev(score4)
std8 = st.stdev(score8)
std12 = st.stdev(score12)
std16 = st.stdev(score16)
std20 = st.stdev(score20)
std24 = st.stdev(score24)
std28 = st.stdev(score28)
std32 = st.stdev(score32)
std = [std0, std4, std8, std12, std16, std20, std24, std28, std32]
print(std)

print(stats.wilcoxon(score0,score4))