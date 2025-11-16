from re import L
import h5py
import torch
import glob
import sys
# sys.path.append('/om4/group/mcdermott/user/imgriff/projects/End-to-end-ASR-Pytorch')
# import src.audio_transforms as audio_transforms
import pickle
import numpy as np
# import typing 
from typing import List, Tuple, Dict, Any, Union, Optional


class SWCPophamCondTestSet2024(torch.utils.data.ConcatDataset):
    # Makes a dataset using pre-paired speech using Popham-style harmonicity manipulations 
    def __init__(self,
                root: str,
                target_harmonicity: Optional[str] = 'harmonic',
                distractor_harmonicity: Optional[Union[str, None]] = 'harmonic') -> None:
        """
        Builds the pytorch hdf5 combined dataset from the files found in the 
        specified root directory. 
        """
        self.all_hdf5_files = glob.glob(f"{root}/*.h5")
        self.target_harmonicity = target_harmonicity
        self.distractor_harmonicity = distractor_harmonicity
        self.all_hdf5_datasets = [H5Dataset(h5_file, self.target_harmonicity, self.distractor_harmonicity) for h5_file in self.all_hdf5_files]

        super().__init__(self.all_hdf5_datasets)

    def class_map(self):
        """
        Loads the mapping between the word IDX and human readable word map. 
        """
        class_map = pickle.load( open("./cv_800_word_label_to_int_dict.pkl", "rb" )) 
        return class_map


class H5Dataset(torch.utils.data.Dataset):
    def __init__(self,
                path: str,
                target_harmonicity: Optional[str] = 'harmonic',
                distractor_harmonicity: Optional[Union[str, None]] = 'harmonic') -> None:
        """
        Builds a pytorch hdf5 dataset
        Args:
            path (str): location of the hdf5 dataset
            target_harmonicity (str): condition for the target speaker, determines key for target and cue in hdf5 file
            distractor_harmonicity (str): condition for the distractor speaker, determines key for distractor in hdf5 file
        """
        self.file_path = path
        self.dataset = None
        self.target_harmonicity = target_harmonicity
        self.distractor_harmonicity = distractor_harmonicity

        with h5py.File(self.file_path, 'r', swmr=True) as file:
            self.dataset_len = len(file['word_int_label'])
        
    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Gets components of the hdf5 file that are used for training
        Args: 
            index (int): index into the hdf5 file
        Returns:
            cue (np.ndarray): cue signal
            target (np.ndarray): target signal
            distractor (np.ndarray): distractor signal
            word_int (np.ndarray): word integer label
            dist_sex_int (np.ndarray): distractor sex integer label
            df_row_ix (np.ndarray): row index corresponding to the parent dataframe 
        """
        if self.dataset is None:
            self.dataset = h5py.File(self.file_path, 'r', swmr=True)
        # set handles for access         
        cue = self.dataset[f"{self.target_harmonicity}_cue_signal"][index]
        target = self.dataset[f"{self.target_harmonicity}_target_signal"][index]
        if self.distractor_harmonicity is not None:
            distractor = self.dataset[f"{self.distractor_harmonicity}_distractor_signal"][index]
        else:
            # set as slience if no distractor
            distractor = None
        word_int = self.dataset['word_int_label'][index]
        dist_sex_int = self.dataset['distractor_sex_int'][index]
        # get the row index of the original dataframe to grab meta in analysis script 
        df_row_ix = self.dataset['orig_df_ix'][index]

        return cue, target, distractor, word_int, dist_sex_int, df_row_ix

    def __len__(self):
        return self.dataset_len
