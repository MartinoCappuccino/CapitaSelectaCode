import os
from tqdm.notebook import tqdm, trange
from typing import List, Tuple, Any, Dict, Set, Optional
from misc import Config

import numpy as np 
from skimage.transform import downscale_local_mean
import SimpleITK as sitk 
# os.environ["PATH"] = "/Users/ivannovikov/Downloads/elastix-5.0.0-mac/bin" + os.pathsep + os.environ["PATH"]
# os.environ["DYLD_LIBRARY_PATH"] = "/Users/ivannovikov/Downloads/elastix-5.0.0-mac/lib"
import elastix
from operator import itemgetter

def dice_score(x, y, eps=1e-5):
    return (2*(x*y).sum()) / ((x+y+eps).sum())

class Model():
    def __init__(
        self,
        config : Config,
        elastix_path : str,
        transformix_path : str,
        metrics = dice_score,
        threshold = 0.5,
        threshold_staple = 0.95,
    ):
        self.__config = config
        self.__el = elastix.ElastixInterface(elastix_path=elastix_path)
        self.transformix_path = transformix_path
        self.metrics = metrics
        self.threshold = threshold
        self.threshold_staple = threshold_staple

    @property
    def config(self):
        return self.__config
    
    @property
    def el(self):
        return self.__el

    def load(self):
        mr_bffe = sitk.ReadImage(os.path.join(self.config.fullpaths_raw[0], "mr_bffe.mhd"))
        mr_bffe = sitk.GetArrayFromImage(mr_bffe)
        self.data = np.zeros(shape=(len(self.config), 2, *mr_bffe.shape)).astype(np.float32)
        for i, path in tqdm(list(enumerate(self.config.fullpaths_raw)), desc="Loading Data"):
            mr_bffe = sitk.ReadImage(os.path.join(path, "mr_bffe.mhd"))
            mr_bffe = sitk.GetArrayFromImage(mr_bffe).astype(np.float32)
            prostaat = sitk.ReadImage(os.path.join(path, "prostaat.mhd"))
            prostaat = sitk.GetArrayFromImage(prostaat).astype(np.float32)
            self.data[i,0] = mr_bffe
            self.data[i,1] = prostaat

    def preprecess_and_save(
        self,
        downscale_factor : int = 4,
        normalize : str = "none"
    ):
        self.data_preprocessed = downscale_local_mean(
            self.data, 
            (1, 1, downscale_factor, downscale_factor, downscale_factor),
        )
        if normalize == "minmax":
            self.data_preprocessed[:,0] -= self.data_preprocessed[:,0].min((1,2,3), keepdims=True)
            self.data_preprocessed[:,0] /= self.data_preprocessed[:,0].max((1,2,3), keepdims=True)
        elif normalize == "zscore":
            self.data_preprocessed[:,0] -= self.data_preprocessed[:,0].mean((1,2,3), keepdims=True)
            self.data_preprocessed[:,0] /= self.data_preprocessed[:,0].std((1,2,3), keepdims=True)
        else:
            assert normalize == "none"
        self.data_preprocessed[:, 1] = (self.data_preprocessed[:, 1] > 0.5).astype(np.float32)
        # self.data_preprocessed[:, 1] 
        for i, path in tqdm(list(enumerate(self.config.fullpaths_preprocessed)), desc="Saving Preprocessed Data"):
            mr_bffe = sitk.GetImageFromArray(self.data_preprocessed[i,0])
            mr_bffe = sitk.Cast(mr_bffe, sitk.sitkFloat32)
            sitk.WriteImage(mr_bffe, os.path.join(path, "mr_bffe.mhd"))
            prostaat = sitk.GetImageFromArray(self.data_preprocessed[i,1])
            prostaat = sitk.Cast(prostaat, sitk.sitkFloat32)
            sitk.WriteImage(prostaat, os.path.join(path, "prostaat.mhd"))

    def register_and_segment(
        self,
        valid_i : int,
        param_i : int,
    ):
        assert valid_i in self.config.valid_indx
        path_fixed = os.path.join(self.config.fullpaths_preprocessed[valid_i], "mr_bffe.mhd")
        path_fixed_mask = os.path.join(self.config.fullpaths_preprocessed[valid_i], "prostaat.mhd")
        path_parameters = self.config.fullpaths_parameters[param_i]
        for train_i in self.config.train_indx:
            path_moving = os.path.join(self.config.fullpaths_preprocessed[train_i], "mr_bffe.mhd")
            path_moving_mask = os.path.join(self.config.fullpaths_preprocessed[train_i], "prostaat.mhd")
            output_dir = os.path.join(
                self.config.folder_results, 
                self.config.basepaths_parameters[param_i][:-4],
                self.config.basepaths[valid_i] + "_as_fixed",
                self.config.basepaths[train_i] + "_as_moving",
            )
            self.el.register(
                fixed_image=path_fixed,
                moving_image=path_moving,
                parameters=[path_parameters],
                output_dir=output_dir,
            )
            
            # depending on if we do 2 or 3 transformations subsequently, the name of the transformation files
            # should be TransformParameters.1.txt and TransformParameters.2.txt respectively
            
            path_transform = os.path.join(output_dir, 'TransformParameters.1.txt')
            with open(path_transform, 'r') as file:
                filedata = file.read()
            # filedata = filedata.replace("(FinalBSplineInterpolationOrder 3)", "(FinalBSplineInterpolationOrder 0)")
            filedata = filedata.replace("(ResultImagePixelType \"short\")", "(ResultImagePixelType \"float\")")
            with open(path_transform, 'w') as file:
                file.write(filedata)

            tr = elastix.TransformixInterface(parameters=path_transform)
            tr.transform_image(path_moving_mask, output_dir=output_dir)

    def run_registration(self):
        for param_i in range(len(self.config.fullpaths_parameters)):
            print("Running with parameters", self.config.basepaths_parameters[param_i])
            for valid_i in tqdm(self.config.valid_indx):
                self.register_and_segment(valid_i, param_i)
        
    def run_staple(self):
        self.segmentations = []
        self.ground_truths = []
        self.scores = []
        for path_parameters in self.config.basepaths_parameters:
            print("Running for parameters", path_parameters)
            temp_1 = os.path.join(self.config.folder_results, path_parameters[:-4])
            res_per_parameters = []
            scores_per_parameters = []
            for path_valid in tqdm(itemgetter(*[*self.config.valid_indx, 0])(self.config.basepaths)[:-1]):
                temp_2 = os.path.join(temp_1, path_valid + "_as_fixed")
                ground_truth = sitk.ReadImage(os.path.join(self.config.folder_preprocessed, path_valid, "prostaat.mhd"))
                ground_truth = sitk.GetArrayFromImage(ground_truth).astype(np.int16)
                self.ground_truths.append(ground_truth)
                seg_stack = []
                for path_train in itemgetter(*[*self.config.train_indx, 0])(self.config.basepaths)[:-1]: 
                    temp_3 = os.path.join(temp_2, path_train + "_as_moving")
                    segmentation = sitk.ReadImage(os.path.join(temp_3, "result.mhd"))
                    segmentation = sitk.GetArrayFromImage(segmentation)
                    segmentation = np.nan_to_num(segmentation, nan=0)
                    segmentation = (segmentation > self.threshold).astype(np.int16)
                    segmentation = sitk.GetImageFromArray(segmentation)
                    seg_stack.append(segmentation)
                staple = sitk.STAPLE(seg_stack, 1.0) 
                staple = sitk.GetArrayFromImage(staple)
                res_per_parameters.append(staple)
                staple = (staple > self.threshold_staple).astype(np.int16)
                score = dice_score(staple, ground_truth)
                scores_per_parameters.append(score)
            self.segmentations.append(res_per_parameters)
            self.scores.append(scores_per_parameters)
