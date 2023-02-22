import elastix
import os
import SimpleITK as sitk
import matplotlib.pyplot as plt
from IndexTracker import IndexTracker

ELASTIX_PATH = os.path.abspath("elastix/elastix.exe")
if not os.path.exists(ELASTIX_PATH):
    raise IOError('Elastix cannot be found, please set the correct ELASTIX_PATH.')
el = elastix.ElastixInterface(elastix_path=ELASTIX_PATH)


TRANSFORMIX_PATH = os.path.abspath("elastix/transformix.exe")
if not os.path.exists(TRANSFORMIX_PATH):
    raise IOError('Transformix cannot be found, please set the correct TRANSFORMIX_PATH.')

if os.path.exists(os.path.abspath("Project/results")) is False:
    os.mkdir(os.path.abspath("Project/results"))

patients = os.listdir(os.path.join("Project/Data"))

dice_scores = {}

for patient in patients:
    if patient[0] == "p" and patient != "p102":
        fixed_image = os.path.join("Project/Data", "p102", "mr_bffe.mhd")
        fixed_label = os.path.join("Project/Data", "p102", "prostaat.mhd")
        moving_image = os.path.join("Project/Data", patient, "mr_bffe.mhd")
        moving_label = os.path.join("Project/Data", patient, "prostaat.mhd")

        parameter0 = os.path.join("Project/CapitaSelectaCode/Registration/Par0001translation.txt")
        parameter1 = os.path.join("Project/CapitaSelectaCode/Registration/Par0001bspline64.txt")
        
        if os.path.exists(os.path.abspath(f"Project/results/{patient}")) is False:
            os.mkdir(os.path.abspath(f"Project/results/{patient}"))

        el.register(
            fixed_image=fixed_image,
            moving_image=moving_image,
            parameters=[parameter0, parameter1],
            output_dir=os.path.abspath(f"Project/results/{patient}")
        )

        tr = elastix.TransformixInterface(parameters=os.path.join("Project/results", patient, "TransformParameters.0.txt"), transformix_path=TRANSFORMIX_PATH)

        if os.path.exists(os.path.abspath(f"Project/results/{patient}/mr_bffe")) is False:
            os.mkdir(os.path.abspath(f"Project/results/{patient}/mr_bffe"))

        if os.path.exists(os.path.abspath(f"Project/results/{patient}/prostaat")) is False:
            os.mkdir(os.path.abspath(f"Project/results/{patient}/prostaat"))
        

        tr.transform_image(moving_image, output_dir=os.path.abspath(f"Project/results/{patient}/mr_bffe"))
        tr.transform_image(moving_label, output_dir=os.path.abspath(f"Project/results/{patient}/prostaat"))

        fixed_image = sitk.GetArrayFromImage(sitk.ReadImage(fixed_image))
        fixed_label = sitk.GetArrayFromImage(sitk.ReadImage(fixed_label))
        transformed_moving_image = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join("Project/results/", patient, "mr_bffe", "result.mhd")))
        transformed_moving_label = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join("Project/results/", patient, "prostaat", "result.mhd")))
        
        TP = 0
        FP = 0
        TN = 0
        FN = 0
        for slice in range(fixed_label.shape[0]):
            for y in range(fixed_label.shape[1]):
                for x in range(fixed_label.shape[2]):
                    if transformed_moving_label[slice, y, x]==1 and fixed_label[slice, y, x]==1:
                        TP += 1
                    elif transformed_moving_label[slice, y, x]==0 and fixed_label[slice, y, x]==0:
                        TN +=1
                    elif transformed_moving_label[slice, y, x]==1 and fixed_label[slice, y, x]==0:
                        FP += 1
                    elif transformed_moving_label[slice, y, x]==0 and fixed_label[slice, y, x]==1:
                        FN +=1

        dice_score = 2*TP / ((TP + FP) + (TP + FN))
        print(dice_score)
        dice_scores[patient] = dice_score

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

print(dice_scores)

