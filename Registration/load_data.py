# import packages
import SimpleITK as sitk
from scrollview import ScrollView
import numpy as np
import matplotlib.pyplot as plt
from IndexTracker import IndexTracker

itk_image = sitk.ReadImage(r"C:\Users\marti\OneDrive - TU Eindhoven\Documenten\Master\Q3\Capita Selecta\Project\Data\p109\mr_bffe.mhd")
image_array = sitk.GetArrayViewFromImage(itk_image)

# print the image's dimensions
print(image_array.shape)
aspect_ratio = [0.488281, 0.488281, 1] # Is the elementspacing in *.mhd files

itk_image_seg = sitk.ReadImage(r'C:\Users\marti\OneDrive - TU Eindhoven\Documenten\Master\Q3\Capita Selecta\Project\Data\p108\mr_bffe.mhd')
segmentation = sitk.GetArrayViewFromImage(itk_image_seg)

# print the image's dimensions
print(segmentation.shape)

# show the image
fig, ax = plt.subplots(1, 2)
tracker1 = IndexTracker(ax[0], image_array)
tracker2 = IndexTracker(ax[1], segmentation)
fig.canvas.mpl_connect('scroll_event', tracker1.onscroll)
fig.canvas.mpl_connect('scroll_event', tracker2.onscroll)

plt.show()