from re import L
import h5py
import torch
import glob
import sys
# sys.path.append('/om4/group/mcdermott/user/imgriff/projects/End-to-end-ASR-Pytorch')
# import src.audio_transforms as audio_transforms
import pickle
import numpy as np


class CommonVoiceWordTask(torch.utils.data.ConcatDataset):
    # Makes a dataset using pre-paired speech and audioset background sounds
    hdf5_glob = '*.hdf5'


    def __init__(self, root, mode='train'):
        """
        Builds the pytorch hdf5 combined dataset from the files found in the 
        specified root directory. 
        """

        if mode == 'train':
            self.all_hdf5_files = glob.glob(root + '/train_*/' + self.hdf5_glob)
        elif mode == 'val':
            self.all_hdf5_files = glob.glob(root + '/validation_*/' + self.hdf5_glob) 
        elif mode == 'test':
            self.all_hdf5_files = glob.glob(root + '/test_*/' + self.hdf5_glob) 

        self.all_hdf5_datasets = [H5Dataset(h5_file, mode) for h5_file in self.all_hdf5_files]

        super().__init__(self.all_hdf5_datasets)

    def class_map(self):
        """
        Loads the mapping between the word IDX and human readable word map. 
        """
        class_map = pickle.load( open("/om2/user/imgriff/datasets/commonvoice_9/en/cv_word_int_label_dict.pkl", "rb" )) 
        return class_map


class H5Dataset(torch.utils.data.Dataset):
    def __init__(self, path, mode):
        """
        Builds a pytorch hdf5 dataset
        Args:
            path (str): location of the hdf5 dataset
        """
        self.file_path = path
        self.dataset = None
        self.mode = mode

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
      
        signal = self.dataset['signal'][index]
        word = self.dataset['label_word_int'][index]

        return signal, word

    def __len__(self):
        return self.dataset_len
