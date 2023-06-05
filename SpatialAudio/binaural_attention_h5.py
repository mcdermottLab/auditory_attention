from re import L
import h5py
import torch
import glob
import sys
# sys.path.append('/om4/group/mcdermott/user/imgriff/projects/End-to-end-ASR-Pytorch')
# import src.audio_transforms as audio_transforms
import pickle
import numpy as np


class BinauralAttentionDataset(torch.utils.data.ConcatDataset):
    # Makes a dataset using pre-paired speech and audioset background sounds
    hdf5_glob = '*.hdf5_chunk000'


    def __init__(self, root, cue_type, task, mode='train', **kwargs):
        """
        Builds the pytorch hdf5 combined dataset from the files found in the 
        specified root directory. 
        """

        if mode == 'train':
            self.all_hdf5_files = glob.glob(root + '/train/' + self.hdf5_glob)
        elif mode == 'val':
            self.all_hdf5_files = glob.glob(root + '/validation/' + self.hdf5_glob) 
        elif mode == 'test':
            self.all_hdf5_files = glob.glob(root + '/test/' + self.hdf5_glob) 

        self.all_hdf5_datasets = [H5Dataset(h5_file, cue_type, task) for h5_file in self.all_hdf5_files]

        super().__init__(self.all_hdf5_datasets)

    def class_map(self):
        """
        Loads the mapping between the word IDX and human readable word map. 
        """
        class_map = pickle.load( open("/om2/user/imgriff/datasets/commonvoice_9/en/cv_word_int_label_dict.pkl", "rb" )) 
        return class_map


class H5Dataset(torch.utils.data.Dataset):
    def __init__(self, path, cue_type, task):
        """
        Builds a pytorch hdf5 dataset
        Args:
            path (str): location of the hdf5 dataset
            task (str): string indicating label keys to return 
        """
        self.file_path = path
        self.dataset = None
        self.task = task

        if cue_type == 'voice_and_location':
            self.cue_key = 'voice_cue_target_loc'
        elif cue_type == 'voice':
            self.cue_key = "voice_cue_center_loc"
        elif cue_type == "location":
            self.cue_key = "loc_cue"

        # TO DO: Implement location and multi-task label handling
        if self.task == 'word':
            self.label_key = 'label_word_int'

        with h5py.File(self.file_path, 'r', swmr=True) as file:
            self.dataset_len = len(file['signal'])

    def __getitem__(self, index):
        """
        Gets components of the hdf5 file that are used for training
        Args: 
            index (int): index into the hdf5 file
        Returns:
            [signal, target] : the training audio (signal) containing the preprocessing
              which may combine the foreground and background speech, and the target idx
              specified by target_keys. 
        """
        if self.dataset is None:
            self.dataset = h5py.File(self.file_path, 'r', swmr=True)

        cue = self.dataset[self.cue_key][index].T
        if np.isnan(cue).all():
            cue[:] = 0
        scene = self.dataset['signal'][index].T
        # TO DO: Implement location and multi-task label handling
        label = self.dataset[self.label_key][index]
        return cue, scene, label

    def __len__(self):
        return self.dataset_len