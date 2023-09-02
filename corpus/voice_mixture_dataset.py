import h5py
import torch
import os 
import numpy as np
from pathlib import Path


class VoiceMixtureDataset(torch.utils.data.ConcatDataset):
    # Makes a dataset by concatenating individual h5-based datasets 
    
    def __init__(self, root, batch_size=1, mode='train', run_mono=False, **kwargs):
        """
        Builds the pytorch hdf5 combined dataset from the files found in the
        specified root directory. 
        """
        self.hdf5_glob = '*.hdf5_chunk000' if with_cue_free else 'noise*.hdf5_chunk000' 
        print(root)
        if mode == 'train':
            self.all_hdf5_files = list(Path(root).glob(f"train/{self.hdf5_glob}"))
            # screen dead files 
            self.all_hdf5_files = [fname for fname in self.all_hdf5_files  if os.path.getsize(fname) > 0]
        elif mode == 'val':
            self.all_hdf5_files =  list(Path(root).glob(f"validation/{self.hdf5_glob}")) 
        elif mode == 'test':
            self.all_hdf5_files = list(Path(root).glob(f"test/{self.hdf5_glob}"))

        # # read files to skip from a file
        files_to_skip = np.load("/om2/user/imgriff/projects/Auditory-Attention/bad_train_file_names.npy")
        # filter bad files from the dataset [generation errors, this is hack to skip bad files]
        
        self.all_hdf5_files = [fname for fname in self.all_hdf5_files if Path(fname).stem not in files_to_skip]

        print(f"{len( self.all_hdf5_files)} files in {mode} concat dataset")
        self.all_hdf5_datasets = [H5Dataset(h5_file, batch_size, run_mono) for h5_file in self.all_hdf5_files]

        super().__init__(self.all_hdf5_datasets)


class H5Dataset(torch.utils.data.Dataset):
    def __init__(self, path, batch_size, run_mono):
        """
        Builds a pytorch hdf5 dataset
        Args:
            path (str): location of the hdf5 dataset
            batch_size (int): int indicating number of samples to draw on each __getitem__ call 
        """
        self.file_path = path
        self.dataset = None
        self.task = task
        self.run_mono = run_mono
        self.batch_size = batch_size
        self.clean_percentage = clean_percentage

        with h5py.File(self.file_path, 'r', swmr=True) as file:
            self.dataset_len = len(file['target']) // self.batch_size

    def azim_elev_to_label(self, azim, elev):
        """ Converts azimuth [0, 360] and elevation [-30, 60] to discrete location classes.
        Addapted to account for negative elevations in training set"""
        if self.skip_negative_elev:
            return np.array(((elev / 10) * 72) + (azim / 5) + 1, dtype=np.int64)
        else:
            return np.array((((elev + 30) / 10) * 72) + (azim / 5), dtype=np.int64)

    def get_competing_row_ix_dict(self):
        """ Pre-identify compliment of row indices for each talker in the dataset.
        Will be used to sample competing talkers for each foreground voice."""
        unique_talker_ints = np.unique(self.dataset['target_talker_id'])
        distractor_row_ix_dict = {talker_int : np.where(self.dataset['target_talker_id'][:] != talker_int)[0] 
                                for talker_int in unique_talker_ints}
        return distractor_row_ix_dict

    def __getitem__(self, index):
        """
        Gets components of the hdf5 file that are used for training
        Args: 
            index (int): index into the hdf5 file
        Returns:
            [foreground, background, foreground_talker_label, foreground_location_label] : 
                        the training foreground and background audio,
                         with talker id and location labels.
        """
        start = index * self.batch_size
        end = start + self.batch_size
        if self.dataset is None:
            self.dataset = h5py.File(self.file_path, 'r', swmr=True)
            self.distractor_row_ix_dict = self.get_competing_row_ix_dict()

        # get foreground voice - transpose to (batch, channel, time)
        foreground = self.dataset['target'][start:end].transpose((0, 2, 1))
        foreground_talker_label = self.dataset['target_talker_id'][start:end]
        foreground_azim = self.dataset['label_loc_target_azim'][start:end]
        foreground_elev = self.dataset['label_loc_target_elev'][start:end]
        foreground_location_label =  self.azim_elev_to_label(foreground_azim, foreground_elev)
        
        # get background voices 
        background_ixs = np.sort([np.random.choice(self.distractor_row_ix_dict[talker_int], size=1)[0] for talker_int in foreground_talker_label])
        background = self.dataset['target'][background_ixs].transpose((0, 2, 1))

        if self.run_mono:
            # average l & r channels
            cue = cue.mean(1).reshape(self.batch_size, -1)
            foreground = foreground.mean(1).reshape(self.batch_size, -1)
            background = background.mean(1).reshape(self.batch_size, -1)


        return foreground, background, foreground_talker_label, foreground_location_label

    def __len__(self):
        return self.dataset_len