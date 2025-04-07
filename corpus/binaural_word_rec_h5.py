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
from pathlib import Path


class BinauralWordRecDataset(torch.utils.data.ConcatDataset):
    # Makes a dataset using pre-paired speech and audioset background sounds
    
    def __init__(self, root, cue_type, task, batch_size=1, mode='train',
                 run_mono=False, mono_sanity_check=False, clean_percentage=0.0, 
                 skip_negative_elev=False,
                 mixture_percentages={'voice_and_location':.33, 'voice_only':.33, "location_only":.33}, **kwargs):
        """
        Builds the pytorch hdf5 combined dataset from the files found in the
        specified root directory. 
        """
        
        self.hdf5_glob = '*.hdf5_chunk000' 
        print(root)

        if mode == 'train':
            self.all_hdf5_files = list(Path(root).glob(f"train_gender_balanced/{self.hdf5_glob}"))
            # screen dead files 
            self.all_hdf5_files = [fname for fname in self.all_hdf5_files  if os.path.getsize(fname) > 0]
        elif mode == 'val':
            self.all_hdf5_files = glob.glob(root + '/validation/' + self.hdf5_glob) 
        elif mode == 'test':
            self.all_hdf5_files = glob.glob(root + '/test/' + self.hdf5_glob) 

        # filter bad files from the dataset on the fly 
        files_to_keep = []
        for file in self.all_hdf5_files:
            with h5py.File(file, 'r', swmr=True) as f:
                if f['sr'][:].all():
                    files_to_keep.append(file)
        self.all_hdf5_files = files_to_keep 
  

        print(f"{len(self.all_hdf5_files)} files in {mode} concat dataset")
        self.all_hdf5_datasets = [H5Dataset(h5_file, cue_type, task, batch_size, run_mono, skip_negative_elev, mono_sanity_check, clean_percentage, mixture_percentages) for h5_file in self.all_hdf5_files]
        super().__init__(self.all_hdf5_datasets)

    def class_map(self):
        """
        Loads the mapping between the word IDX and human readable word map. 
        """
        class_map = pickle.load( open("cv_800_word_label_to_int_dict.pkl", "rb" )) 
        return class_map


class H5Dataset(torch.utils.data.Dataset):
    def __init__(self, path, scene_type, task, batch_size, run_mono, skip_negative_elev=False, mono_sanity_check=False,
                      clean_percentage=0.0, mixture_percentages={'voice_and_location':.50, 'voice_only':.50},
                      **kwargs):
        """
        Builds a pytorch hdf5 dataset for of binaural word recognition dataset
        This class handles batching, returning batches of cues and scenes.
        v06 uses separate target bg_scenes to be mixed on the fly at a chosen SNR.
        All cues are "voice_cue_target_loc" for v06: the target voice at the target location.
        The "voice only" condition is achieved by using scenes where all signals are co-located. 

        Parameters:
            path (str): location of the hdf5 dataset
            scene_type (str): string indicating whether to use both co-located and normal scenes, or just normal scenes.
            task (str): string indicating label keys to return - word, location, or both
            batch_size (int): batch size
            run_mono (bool): whether to return mono signals or stereo signals (default)
            skip_negative_elev (bool): whether to skip negative elevation labels (default False)
            mono_sanity_check (bool): whether to use only left channel for both channels (default False)
            mixture_percentages (dict): dictionary of percentages for each scene type in the mixed scene condition.
                The keys are 'voice_and_location' and 'voice_only'. The values are floats that sum to 1.0.
        Returns:
            cue, target, background, label
        """
        self.file_path = Path(path).as_posix()
        self.dataset = None
        self.task = task
        self.run_mono = run_mono
        self.batch_size = batch_size
        self.skip_negative_elev = skip_negative_elev
        self.scene_type = scene_type
        self.clean_percentage = clean_percentage
        self.mono_sanity_check = mono_sanity_check
        if scene_type == "mixed":
            self.voice_and_loc_size = int(batch_size * mixture_percentages['voice_and_location'])
            self.voice_only_percent_size = int(batch_size * mixture_percentages['voice_only'])

        with h5py.File(self.file_path, 'r', swmr=True) as file:
            self.safe_ixs = np.where(file['n_speech_distractors'][:] == 0)[0]
            self.dataset_len = len(self.safe_ixs) // self.batch_size

    def __getitem__(self, index):
        """
        Returns examples from the hdf5 file.
        Args: 
            index (int): index into the hdf5 file
        Returns:
            None (np.array): instead of cue for compatibility with modules
            target (np.array): the target audio 
            background (None):the background audio 
            label (np.array): the label for the example.
        """
        if self.dataset is None:
            self.dataset = h5py.File(self.file_path, 'r', swmr=True)

        index = self.safe_ixs[index]
        target = self.dataset['target'][index].transpose((1, 0))

        if self.scene_type == 'mixed':
            # get cut sizes for each condition
            bg_key = np.random.choice(['bg_scene', 'bg_scene_co_located'])
            background = self.dataset[bg_key][index].transpose((1, 0))
        else:
            background = self.dataset['bg_scene'][index].transpose((1, 0))

        label = self.dataset['word_int'][index]
  
        if self.clean_percentage > 0.0:
            if np.random.rand() > self.clean_percentage:
                background[:] = 0 

        if self.run_mono:
            # Just take single channel
            target = target[:,0,:].reshape(self.batch_size, -1)
            background = background[:,0,:].reshape(self.batch_size, -1)
            
        if self.mono_sanity_check:
            # running diotic. Sum channels then copy to both channels
            target = np.sum(target, axis=1, keepdims=True).repeat(2, axis=1)
            background = np.sum(background, axis=1, keepdims=True).repeat(2, axis=1)
            
        return None, target, background, label

    def __len__(self):
        return self.dataset_len