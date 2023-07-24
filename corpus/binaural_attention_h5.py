from re import L
import h5py
import torch
import glob
import sys
import os 
# sys.path.append('/om4/group/mcdermott/user/imgriff/projects/End-to-end-ASR-Pytorch')
# import src.audio_transforms as audio_transforms
import pickle
import numpy as np


class BinauralAttentionDataset(torch.utils.data.ConcatDataset):
    # Makes a dataset using pre-paired speech and audioset background sounds
    
    def __init__(self, root, cue_type, task, batch_size=1, skip_negative_elev=False, mode='train', with_cue_free=False, **kwargs):
        """
        Builds the pytorch hdf5 combined dataset from the files found in the
        specified root directory. 
        """
        self.hdf5_glob = '*.hdf5_chunk000' if with_cue_free else 'noise*.hdf5_chunk000' 

        if mode == 'train':
            self.all_hdf5_files = glob.glob(root + '/train/' + self.hdf5_glob)
            # screen dead files 
            self.all_hdf5_files = [fname for fname in self.all_hdf5_files  if os.path.getsize(fname) > 0]
        elif mode == 'val':
            self.all_hdf5_files = glob.glob(root + '/validation/' + self.hdf5_glob) 
        elif mode == 'test':
            self.all_hdf5_files = glob.glob(root + '/test/' + self.hdf5_glob) 

        # read files to skip from a file
        with open(root + '/bad_files.txt', 'r') as f:
            files_to_skip = [line.strip().split('/')[-1] for line in f.readlines()]
        # filter bad files from the dataset
        self.all_hdf5_files = [fname for fname in self.all_hdf5_files if fname.split('/')[-1] not in files_to_skip]

        print(f"{len( self.all_hdf5_files)} files in {mode} concat dataset")
        self.all_hdf5_datasets = [H5Dataset(h5_file, cue_type, task, batch_size, skip_negative_elev) for h5_file in self.all_hdf5_files]

        super().__init__(self.all_hdf5_datasets)

    def class_map(self):
        """
        Loads the mapping between the word IDX and human readable word map. 
        """
        class_map = pickle.load( open("/om2/user/imgriff/datasets/commonvoice_9/en/cv_800_word_label_to_int_dict.pkl", "rb" )) 
        return class_map


class H5Dataset(torch.utils.data.Dataset):
    def __init__(self, path, cue_type, task, batch_size, skip_negative_elev=False):
        """
        Builds a pytorch hdf5 dataset
        Args:
            path (str): location of the hdf5 dataset
            task (str): string indicating label keys to return 
        """
        self.file_path = path
        self.dataset = None
        self.task = task
        self.batch_size = batch_size
        self.skip_negative_elev = skip_negative_elev
        # if self.skip_negative_elev:
        #     print("Skipping negative elevations")
        # else:
        #     print("Including negative elevations")

        if cue_type == 'voice_and_location':
            self.cue_key = 'voice_cue_target_loc'
        elif cue_type == 'voice':
            self.cue_key = "voice_cue_center_loc"
        elif cue_type == "location":
            self.cue_key = "loc_cue"
        elif cue_type == "mixed":
            self.cue_key = "mixed_cue"

        if self.task == 'word':
            self.label_key = 'word_int'
        elif self.task == 'location':
            self.label_key = ('label_loc_target_azim', 'label_loc_target_elev')
        elif self.task == 'word_and_location':
            self.label_key = ('word_int', 'label_loc_target_azim', 'label_loc_target_elev')

        with h5py.File(self.file_path, 'r', swmr=True) as file:
            self.dataset_len = len(file['target']) // self.batch_size

    def azim_elev_to_label(self, azim, elev):
        if self.skip_negative_elev:
            return np.array(((elev / 10) * 72) + (azim / 5) + 1, dtype=np.int64)
        else:
            return np.array((((elev + 30) / 10) * 72) + (azim / 5), dtype=np.int64)

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
        start = index * self.batch_size
        end = start + self.batch_size
        if self.dataset is None:
            self.dataset = h5py.File(self.file_path, 'r', swmr=True)

        if self.cue_key == 'mixed_cue':
            cut1 = start + (self.batch_size // 3)
            cut2 = start + ((self.batch_size // 3) * 2)
            loc_cue = self.dataset['loc_cue']
            voice_cue = self.dataset['voice_cue_center_loc']
            both_cue = self.dataset['voice_cue_target_loc']
            cues1 = loc_cue[start:cut1].transpose((0, 2, 1))
            cues2 = voice_cue[cut1:cut2].transpose((0, 2, 1))
            cues3 = both_cue[cut2:end].transpose((0, 2, 1))
            cue = np.concatenate((cues1, cues2, cues3), axis=0)
        else:
            cue = self.dataset[self.cue_key][start:end].transpose((0, 2, 1))
        if np.isnan(cue).all():
            cue[:] = 0
        foreground = self.dataset['target'][start:end].transpose((0, 2, 1))
        background = self.dataset['bg_scene'][start:end].transpose((0, 2, 1))

        # if self.skip_negative_elev and self.dataset['label_loc_target_elev'][start:end] < 0:
        #     return None, None, None, None

        if self.task == 'word':
            label = self.dataset[self.label_key][start:end]
        elif self.task == 'location':
            azim = self.dataset[self.label_key[0]][start:end]
            elev = self.dataset[self.label_key[1]][start:end]
            label = []
            for azim, elev in zip(azim, elev):
                label.append(self.azim_elev_to_label(azim, elev))
        else:
            word = self.dataset[self.label_key[0]][start:end]
            azim = self.dataset[self.label_key[1]][start:end]
            elev = self.dataset[self.label_key[2]][start:end]
            loc = self.azim_elev_to_label(azim, elev)
            label = np.stack((word, loc), axis=1)
        return cue, foreground, background, label

    def __len__(self):
        return self.dataset_len