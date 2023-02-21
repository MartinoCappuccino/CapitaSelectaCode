import elastix
import os
import SimpleITK as sitk
import matplotlib.pyplot as plt

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

for patient in patients:
    if patient[0] == "p" and patient != "p135":
        fixed_image = os.path.join("Project/Data", "p135", "mr_bffe.mhd")
        fixed_label = os.path.join("Project/Data", "p135", "prostaat.mhd")
        moving_image = os.path.join("Project/Data", patient, "mr_bffe.mhd")
        moving_label = os.path.join("Project/Data", patient, "prostaat.mhd")

        parameters = os.path.join("Project/CapitaSelectaCode/PAR0001bspline64.txt")
        
        if os.path.exists(os.path.abspath(f"Project/results/{patient}")) is False:
            os.mkdir(os.path.abspath(f"Project/results/{patient}"))

        el.register(
            fixed_image=fixed_image,
            moving_image=moving_image,
            parameters=[parameters],
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

        fig, ax = plt.subplots(1, 4, figsize=(20, 5))
        ax[0].imshow(fixed_image[43, :, :], cmap='gray')
        ax[0].set_title('Fixed image')
        ax[1].imshow(fixed_label[43, :, :], cmap='gray')
        ax[1].set_title('Fixed label')
        ax[2].imshow(transformed_moving_image[43, :, :], cmap='gray')
        ax[2].set_title('Transformed\nmoving image')
        ax[3].imshow(transformed_moving_label[43, :, :], cmap='gray')
        ax[3].set_title('Transformed\nmoving label')
        [x.set_axis_off() for x in ax]
        plt.show()

        break

