from typing import List, Tuple, Any, Dict, Set, Optional
from sklearn.model_selection import train_test_split
import os
from operator import itemgetter
import datetime
os.pathsep="/"

def _is_valid_name(name):
    try:
        n = int(name[1:])
    except:
        return False
    return (name[0] == 'p') and (len(name) == 4)

class Config():
    def __init__(
            self, 
            folder_raw : str, 
            folder_preprocessed : str,
            parameter_folder : str,
            parameters: list,
            folder_results : str,
            seed = 0,
            train_size = 12,
            train_indx = None,
            valid_indx = None,
        ):
            assert os.path.exists(folder_raw)
            
            if not os.path.exists(folder_preprocessed):
                os.mkdir(folder_preprocessed)
            assert os.path.exists(folder_preprocessed)

            assert os.path.exists(parameter_folder)
            
            if not os.path.exists(folder_results):
                os.mkdir(folder_results)
            assert os.path.exists(folder_results)

            self.__folder_raw = folder_raw
            self.__folder_preprocessed = folder_preprocessed
            self.__parameter_folder = parameter_folder
            self.__parameters = parameters
            self.__folder_results = folder_results
            
            self.__patients = list(sorted(filter(_is_valid_name, os.listdir(self.folder_raw))))
            self.__fullpaths_raw = list(map(lambda path : os.path.join(self.folder_raw, path), self.patients))
            self.__fullpaths_preprocessed = list(map(lambda path : os.path.join(self.folder_preprocessed, path), self.patients))
            
            for path in self.fullpaths_preprocessed:
                if not os.path.exists(path):
                    os.mkdir(path)
            self.set_train_valid_indx(seed, train_size, train_indx, valid_indx)

            self.__fullpaths_results = []

            self.__now = datetime.datetime.now().strftime("%y-%m-%d_%H-%M-%S")
            if not os.path.exists(os.path.join(folder_results, self.now)):
                os.mkdir(os.path.join(self.folder_results, self.now))
            for valid_i in self.valid_indx:
                if not os.path.exists(os.path.join(folder_results, self.now, self.patients[valid_i])):
                    os.mkdir(os.path.join(self.folder_results, self.now, self.patients[valid_i]))
                for patient in self.patients:
                    if not os.path.exists(os.path.join(folder_results, self.now, self.patients[valid_i], patient)) and patient != self.patients[valid_i]:
                        os.mkdir(os.path.join(self.folder_results, self.now, self.patients[valid_i], patient))
                        self.fullpaths_results.append(os.path.join(self.folder_results, self.now, self.patients[valid_i], patient))

            # self.__basepaths_parameters_per_group = []
            # self.__fullpaths_parameters_per_group = []

            # for i in range(len(self.basepaths_parameter_groups)):
            #     path_parameter_group = self.basepaths_parameter_groups[i]
            #     fullpath_parameter_group = self.fullpaths_parameter_groups[i]
            #     if not os.path.isdir(fullpath_parameter_group):
            #         continue
            #     # Creating results folders
            #     temp_1 = os.path.join(self.folder_results, path_parameter_group)
            #     if not os.path.exists(temp_1):
            #         os.mkdir(temp_1)
            #     for path_valid in itemgetter(*[*self.valid_indx, 0])(self.basepaths)[:-1]:
            #         temp_2 = os.path.join(temp_1, path_valid + "_as_fixed")
            #         if not os.path.exists(temp_2):
            #             os.mkdir(temp_2)
            #         for path_train in itemgetter(*[*self.train_indx, 0])(self.basepaths)[:-1]:
            #             temp_3 = os.path.join(temp_2, path_train + "_as_moving")
            #             if not os.path.exists(temp_3):
            #                 os.mkdir(temp_3)
            #     # Storing parameter files per group (SORTED BY NAME)
            #     paths_parameters = list(sorted(
            #         filter(
            #         lambda path : path[-4:] == ".txt", 
            #         os.listdir(fullpath_parameter_group))
            #     ))
            #     fullpaths_parameters = list(map(
            #         lambda path : os.path.join(fullpath_parameter_group, path),
            #         paths_parameters,
            #     ))
            #     self.basepaths_parameters_per_group.append(paths_parameters)
            #     self.fullpaths_parameters_per_group.append(fullpaths_parameters)

    def __len__(self):
        return len(self.patients)
    
    def set_train_valid_indx(
        self,
        seed : int = 0,
        train_size : int = 12,
        train_indx : Optional[List[str]] = None,
        valid_indx : Optional[List[str]] = None,
    ):
        if train_indx is not None:
            if valid_indx is not None:
                assert len(set(valid_indx) & set(train_indx)) == 0
            else:
                valid_indx = list(set(range(len(self))) - set(train_indx))
        elif valid_indx is not None:
            train_indx = list(set(range(len(self))) - set(valid_indx))
        else:
            train_indx, valid_indx = train_test_split(range(len(self)), random_state=seed, train_size=train_size)
        self.__train_indx = train_indx
        self.__valid_indx = valid_indx
        
    @property
    def folder_raw(self):
        return self.__folder_raw
    
    @property
    def folder_preprocessed(self):
        return self.__folder_preprocessed
    
    @property
    def parameter_folder(self):
        return self.__parameter_folder
    
    @property
    def parameters(self):
        return self.__parameters
    
    @property
    def folder_results(self):
        return self.__folder_results
    
    @property
    def patients(self):
        return self.__patients
    
    @property
    def fullpaths_raw(self):
        return self.__fullpaths_raw
    
    @property
    def fullpaths_preprocessed(self):
        return self.__fullpaths_preprocessed
    
    @property
    def fullpaths_results(self):
        return self.__fullpaths_results

    @property
    def train_indx(self):
        return self.__train_indx
    
    @property
    def valid_indx(self):
        return self.__valid_indx
    
    @property
    def now(self):
        return self.__now
