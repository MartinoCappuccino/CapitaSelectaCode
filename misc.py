from typing import List, Tuple, Any, Dict, Set, Optional
from sklearn.model_selection import train_test_split
import os
from operator import itemgetter

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
            folder_parameters : str,
            folder_results : str,
            seed = 0,
            train_size = 12,
            train_indx = None,
            valid_indx = None,
        ):
            assert os.path.exists(folder_raw)
            assert os.path.exists(folder_preprocessed)
            assert os.path.exists(folder_parameters)
            assert os.path.exists(folder_results)

            self.__folder_raw = folder_raw
            self.__folder_preprocessed = folder_preprocessed
            self.__folder_parameters = folder_parameters
            self.__folder_results = folder_results
            self.__basepaths = list(sorted(filter(_is_valid_name, os.listdir(self.folder_raw))))
            self.__fullpaths_raw = list(map(lambda path : os.path.join(self.folder_raw, path), self.basepaths))
            self.__fullpaths_preprocessed = list(map(
                lambda path : os.path.join(self.folder_preprocessed, path), 
                self.basepaths
            ))
            for path in self.fullpaths_preprocessed:
                if not os.path.exists(path):
                    os.mkdir(path)
            self.set_train_valid_indx(seed, train_size, train_indx, valid_indx)

            self.__basepaths_parameters = list(sorted(os.listdir(self.folder_parameters)))
            self.__fullpaths_parameters = list(map(
                lambda path : os.path.join(self.folder_parameters, path),
                self.basepaths_parameters,
            ))
            for path_parameters in self.basepaths_parameters:
                assert path_parameters[-4:] == ".txt"
                temp_1 = os.path.join(self.folder_results, path_parameters[:-4])
                if not os.path.exists(temp_1):
                    os.mkdir(temp_1)
                for path_valid in itemgetter(*[*self.valid_indx, 0])(self.basepaths)[:-1]:
                    temp_2 = os.path.join(temp_1, path_valid + "_as_fixed")
                    if not os.path.exists(temp_2):
                        os.mkdir(temp_2)
                    for path_train in itemgetter(*[*self.train_indx, 0])(self.basepaths)[:-1]:
                        temp_3 = os.path.join(temp_2, path_train + "_as_moving")
                        if not os.path.exists(temp_3):
                            os.mkdir(temp_3)

    def __len__(self):
        return len(self.basepaths)
    
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
    def folder_parameters(self):
        return self.__folder_parameters
    
    @property
    def folder_results(self):
        return self.__folder_results
    
    @property
    def basepaths(self):
        return self.__basepaths
    
    @property
    def fullpaths_raw(self):
        return self.__fullpaths_raw
    
    @property
    def fullpaths_preprocessed(self):
        return self.__fullpaths_preprocessed
    
    @property
    def basepaths_parameters(self):
        return self.__basepaths_parameters
    
    @property
    def fullpaths_parameters(self):
        return self.__fullpaths_parameters
    
    @property
    def train_indx(self):
        return self.__train_indx
    
    @property
    def valid_indx(self):
        return self.__valid_indx