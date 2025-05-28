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


class BinauralAttentionDataset(torch.utils.data.ConcatDataset):
    # Makes a dataset using pre-paired speech and audioset background sounds
    
    def __init__(self, root, cue_type, task, batch_size=1, skip_negative_elev=False, mode='train',
                 with_cue_free=False, run_mono=False, mono_sanity_check=False, clean_percentage=0.0, v05=False, v06=False,
                 gender_balanced=False,
                 gender_balanced_4M=False,
                 mostly_same_dist=False,
                 cue_free_percentage=None,
                 mixture_percentages={'voice_and_location':.33, 'voice_only':.33, "location_only":.33}, **kwargs):
        """
        Builds the pytorch hdf5 combined dataset from the files found in the
        specified root directory. 
        """
        self.v05 = v05
        self.v06 = v06
        
        if v05 or "v05" in root:
            self.v05 = True
            print("Using v05 dataset")
        elif v06 or "v06" in root:
            self.v06 = True
            print("Using v06 dataset")
        self.hdf5_glob = '*.hdf5_chunk000' 
        # self.hdf5_glob = '*.hdf5_chunk000' if with_cue_free or self.v05 else 'noise*.hdf5_chunk000' 
        print(root)

        if cue_free_percentage is not None and mode == 'train':
            no_cue_on_clean = True
            print(f"Using {cue_free_percentage} cue free data")
            clean_percentage = cue_free_percentage
        else:
            no_cue_on_clean = False

        if mode == 'train':
            if gender_balanced or gender_balanced_4M:
                if gender_balanced_4M or 'v08' in root:
                    print("Using gender balanced training 4M set")
                    self.all_hdf5_files = list(Path(root).glob(f"train_gender_balanced/{self.hdf5_glob}"))
                elif gender_balanced_4M:
                    print("Using gender balanced training 4M set")
                    self.all_hdf5_files = list(Path(root).glob(f"train_gender_balanced_4M/{self.hdf5_glob}"))
                else:
                    print("Using gender balanced training set")
                    self.all_hdf5_files = list(Path(root).glob(f"train_gender_balanced/{self.hdf5_glob}"))
            elif mostly_same_dist:
                print("Using 70% same-sex distractor training set")
                self.all_hdf5_files = list(Path(root).glob(f"train_70p_1dist_same_sex/{self.hdf5_glob}"))
            else:
                self.all_hdf5_files = list(Path(root).glob(f"train/{self.hdf5_glob}"))
            # screen dead files 
            self.all_hdf5_files = [fname for fname in self.all_hdf5_files  if os.path.getsize(fname) > 0]
        elif mode == 'val':
            self.all_hdf5_files = glob.glob(root + '/validation/' + self.hdf5_glob) 
        elif mode == 'test':
            self.all_hdf5_files = glob.glob(root + '/test/' + self.hdf5_glob) 

        # # read files to skip from a file
        # with open(root + '/bad_files.txt', 'r') as f:
        #     files_to_skip = [line.strip().split('/')[-1] for line in f.readlines()]
        if not self.v05 and not self.v06:
            # filter bad files from the dataset
            files_to_skip = np.load("/om2/user/imgriff/projects/Auditory-Attention/bad_train_file_names.npy")
            self.all_hdf5_files = [fname for fname in self.all_hdf5_files if Path(fname).stem not in files_to_skip]
        elif self.v05 or self.v06:
            # filter bad files from the dataset on the fly 
            files_to_keep = []
            for file in self.all_hdf5_files:
                with h5py.File(file, 'r', swmr=True) as f:
                    if f['sr'][:].all():
                        files_to_keep.append(file)
            self.all_hdf5_files = files_to_keep 
                
        print(f"cue type: {cue_type}")
        if cue_type == 'mixed':
            print(f"{mixture_percentages=}")

        print(f"{len(self.all_hdf5_files)} files in {mode} concat dataset")
        if self.v05:
            dataset_class = H5DatasetV05
        elif self.v06:
            dataset_class = H5DatasetV06
        else:
            dataset_class = H5Dataset
        self.all_hdf5_datasets = [dataset_class(h5_file, cue_type, task, batch_size, run_mono, skip_negative_elev, mono_sanity_check, clean_percentage, mixture_percentages, no_cue_on_clean) for h5_file in self.all_hdf5_files]

        super().__init__(self.all_hdf5_datasets)

    def class_map(self):
        """
        Loads the mapping between the word IDX and human readable word map. 
        """
        class_map = pickle.load( open("./cv_800_word_label_to_int_dict.pkl", "rb" )) 
        return class_map


class H5Dataset(torch.utils.data.Dataset):
    def __init__(self, path, cue_type, task, batch_size, run_mono, skip_negative_elev=False, mono_sanity_check=False, 
                clean_percentage=0.0, mixture_percentages={'voice_and_location':.33, 'voice_only':.33, "location_only":.33}, no_cue_on_clean=False):
        """
        Builds a pytorch hdf5 dataset
        Args:
            path (str): location of the hdf5 dataset
            task (str): string indicating label keys to return 
        """
        self.file_path = Path(path).as_posix()
        self.dataset = None
        self.task = task
        self.run_mono = run_mono
        self.batch_size = batch_size
        self.skip_negative_elev = skip_negative_elev
        self.mono_sanity_check = mono_sanity_check
        self.clean_percentage = clean_percentage
        self.no_cue_on_clean = no_cue_on_clean
        self.batch_ixs = np.arange(self.batch_size)
        if self.skip_negative_elev:
            print("Skipping negative elevations")
        else:
            print("Including negative elevations")

        if "v02" in self.file_path:
            self.voice_key = "voice_cue_rand_loc"
        elif "v03" in self.file_path:
            self.voice_key = "voice_cue_center_loc"
        else:
            self.voice_key = "voice_cue_target_loc"

        if cue_type == 'voice_and_location':
            self.cue_key = 'voice_cue_target_loc'
        elif cue_type == 'voice':
            self.cue_key = self.voice_key
        elif cue_type == "location":
            self.cue_key = "loc_cue"
        elif cue_type == "mixed":
            self.cue_key = "mixed_cue"
            self.voice_and_loc_size = int(batch_size * mixture_percentages['voice_and_location'])
            self.voice_only_percent_size = int(batch_size * mixture_percentages['voice_only'])
            self.location_only_percent_size = int(batch_size * mixture_percentages['location_only']) 

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
            # + 40 is so lowest elevation is 0 in label index
            return np.array((((elev + 40) / 10) * 72) + (azim / 5), dtype=np.int64)

    def __getitem__(self, index):
        """
        Returns examples from the hdf5 file.
        Args: 
            index (int): index into the hdf5 file
        Returns:
            cue (np.array): the cue audio 
            foreground (np.array): the target audio 
            background (np.array): the distractor audio 
            label (np.array): the label for the batch
        """
        start = index * self.batch_size
        end = start + self.batch_size
        if self.dataset is None:
            self.dataset = h5py.File(self.file_path, 'r', swmr=True)
        # Updates on 10/11/2023 - use "both_cue" for 2/3 of mixed cue training
        # Will use different backgrounds for voice-only condition cue type
        if self.cue_key == 'mixed_cue':
            # get cut sizes for each condition
            loc_only_ix = start + self.location_only_percent_size
            normal_ix = loc_only_ix + self.voice_and_loc_size
            co_located_ix = normal_ix + self.voice_only_percent_size
            # cue handles 
            loc_cue = self.dataset['loc_cue']
            # use both cue for voice + location and voice only conditions
            both_cue = self.dataset['voice_cue_target_loc']
            cues1 = loc_cue[start : loc_only_ix].transpose((0, 2, 1))
            cues2 = both_cue[loc_only_ix : normal_ix].transpose((0, 2, 1)) 
            cues3 = both_cue[normal_ix : end].transpose((0, 2, 1)) 
            # first 1/3 location, last 2/3 both
            cue = np.concatenate((cues1, cues2, cues3), axis=0)
            ## get backgrounds for mixed cue condition
            # get backgrounds for location only and voice + location conditions
            bg_1 = self.dataset['bg_scene'][start:loc_only_ix].transpose((0, 2, 1))
            bg_2 = self.dataset['bg_scene'][loc_only_ix:normal_ix].transpose((0, 2, 1))
            bg_3 = self.dataset['bg_scene_co_located'][normal_ix: end].transpose((0, 2, 1))
            # first 2/3 normal, last 1/3 co-located
            background = np.concatenate((bg_1, bg_2, bg_3), axis=0) 

        elif self.cue_key == 'voice':
            cue = self.dataset[self.cue_key][start:end].transpose((0, 2, 1))
            background = self.dataset['bg_scene_co_located'][start:end].transpose((0, 2, 1))
        else:
            cue = self.dataset[self.cue_key][start:end].transpose((0, 2, 1))
            background = self.dataset['bg_scene'][start:end].transpose((0, 2, 1))

        if np.isnan(cue).all():
            cue[:] = 0
        foreground = self.dataset['target'][start:end].transpose((0, 2, 1))

        if self.clean_percentage > 0.0:
            num_clean = int(self.clean_percentage * self.batch_size)
            clean_idx = np.random.choice(self.batch_ixs, num_clean, replace=False)
            background[clean_idx, :, :] = 0
            if self.no_cue_on_clean:
                cue[clean_idx, :, :] = 0

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


        if self.run_mono:
            # average l & r channels
            cue = cue.mean(1).reshape(self.batch_size, -1)
            foreground = foreground.mean(1).reshape(self.batch_size, -1)
            background = background.mean(1).reshape(self.batch_size, -1)
            
        if self.mono_sanity_check:
            # use only left channel for both channels
            cue[:,1,:] = cue[:,0,:]
            foreground[:,1,:] = foreground[:,0,:]
            background[:,1,:] = background[:,0,:]
            
        return cue, foreground, background, label

    def __len__(self):
        return self.dataset_len
    


class H5DatasetV05(torch.utils.data.Dataset):
    def __init__(self, path, scene_type, task, batch_size, run_mono, skip_negative_elev=False, mono_sanity_check=False,  clean_percentage=0.0, mixture_percentages={'voice_and_location':.50, 'voice_only':.50}, **kwargs):
        """
        Builds a pytorch hdf5 dataset for v05 of binaural attention dataset
        This class handles batching, returning batches of cues and scenes.
        v05 uses pre-mixed scenes "signal" and "signal_co_located" for the mixed scene condition.
        All cues are "voice_cue_target_loc" for v05: the target voice at the target location.
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
            cue, scene, None, label - None returned instead of background for compatability with past versions. 
        """
        self.file_path = Path(path).as_posix()
        self.dataset = None
        self.task = task
        self.run_mono = run_mono
        self.batch_size = batch_size
        self.skip_negative_elev = skip_negative_elev
        self.mono_sanity_check = mono_sanity_check
        self.batch_ixs = np.arange(self.batch_size)

        self.cue_key = 'voice_cue_target_loc'
        self.scene_type = scene_type
        if scene_type == "mixed":
            self.voice_and_loc_size = int(batch_size * mixture_percentages['voice_and_location'])
            self.voice_only_percent_size = int(batch_size * mixture_percentages['voice_only'])

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
        Returns examples from the hdf5 file.
        Args: 
            index (int): index into the hdf5 file
        Returns:
            cue (np.array): the cue audio 
            scene (np.array): the scene audio 
            None (None): None - instead of background for compatability with past versions
            label (np.array): the label for the batch.
        """
        start = index * self.batch_size
        end = start + self.batch_size
        if self.dataset is None:
            self.dataset = h5py.File(self.file_path, 'r', swmr=True)

        cue = self.dataset['voice_cue_target_loc'][start:end].transpose((0, 2, 1))

        if self.scene_type == 'mixed':
            # get cut sizes for each condition
            normal_ix = start + self.voice_and_loc_size
            co_located_ix = normal_ix + self.voice_only_percent_size    
            ## get backgrounds for mixed cue condition
            # get backgrounds for location only and voice + location conditions
            scene_1 = self.dataset['signal'][start : normal_ix].transpose((0, 2, 1))
            scene_2 = self.dataset['signal_co_located'][normal_ix : co_located_ix].transpose((0, 2, 1))
            # first 2/3 normal, last 1/3 co-located
            scene = np.concatenate((scene_1, scene_2), axis=0) 
        else:
            scene = self.dataset['signal'][start:end].transpose((0, 2, 1))

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

        if self.run_mono:
            # average l & r channels
            cue = cue.mean(1).reshape(self.batch_size, -1)
            scene = scene.mean(1).reshape(self.batch_size, -1)
            
        if self.mono_sanity_check:
            # use only left channel for both channels
            cue[:,1,:] = cue[:,0,:]
            scene[:,1,:] = scene[:,0,:]
            scene[:,1,:] = scene[:,0,:]
            
        return cue, scene, None, label

    def __len__(self):
        return self.dataset_len


class H5DatasetV06(torch.utils.data.Dataset):
    def __init__(self, path, scene_type, task, batch_size, run_mono, skip_negative_elev=False, mono_sanity_check=False,  clean_percentage=0.0, mixture_percentages={'voice_and_location':.50, 'voice_only':.50}, no_cue_on_clean=False, **kwargs):
        """
        Builds a pytorch hdf5 dataset for v06 of binaural attention dataset
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
        self.mono_sanity_check = mono_sanity_check
        self.clean_percentage = clean_percentage
        self.no_cue_on_clean = no_cue_on_clean
        # set logic for sampling cue-free examples
        self.batch_ixs = np.arange(self.batch_size)
        self.ix_probs = torch.tensor([1/batch_size] * batch_size) # uniform ix likelihoods 
        self.n_to_sample = int(self.clean_percentage * self.batch_size)
      
        self.cue_key = 'voice_cue_target_loc'
        self.scene_type = scene_type
        if scene_type == "mixed":
            self.voice_and_loc_size = int(batch_size * mixture_percentages['voice_and_location'])
            self.voice_only_percent_size = int(batch_size * mixture_percentages['voice_only'])
        if self.task == 'word':
            self.label_key = 'word_int'
        elif self.task == 'location':
            self.label_key = ['label_loc_target_azim', 'label_loc_target_elev']
        elif self.task == 'word_and_location':
            self.label_key = ('word_int', 'label_loc_target_azim', 'label_loc_target_elev')

        with h5py.File(self.file_path, 'r', swmr=True) as file:
            self.dataset_len = len(file['target']) // self.batch_size

    def azim_elev_to_label(self, azim, elev):
        if self.skip_negative_elev:
            return np.array(((elev / 10) * 72) + (azim / 5), dtype=np.int64)
        else:
            # + 40 is so lowest elevation is 0 in label index
            return np.array((((elev + 40) / 10) * 72) + (azim / 5), dtype=np.int64)

    def label_to_azim_elev(self, label):
        """
        """
        elev = np.array((label // 72) * 10)
        azim = np.array((label % 72) * 5)
        return np.array(azim).astype(float), np.array(elev).astype(float)

    def __getitem__(self, index):
        """
        Returns examples from the hdf5 file.
        Args: 
            index (int): index into the hdf5 file
        Returns:
            cue (np.array): the cue audio 
            scene (np.array): the scene audio 
            None (None): None - instead of background for compatability with past versions
            label (np.array): the label for the batch.
        """
        start = index * self.batch_size
        end = start + self.batch_size
        if self.dataset is None:
            self.dataset = h5py.File(self.file_path, 'r', swmr=True)

        cue = self.dataset['voice_cue_target_loc'][start:end].transpose((0, 2, 1))
        target = self.dataset['target'][start:end].transpose((0, 2, 1))

        if self.scene_type == 'mixed':
            # get cut sizes for each condition
            normal_ix = start + self.voice_and_loc_size
            co_located_ix = normal_ix + self.voice_only_percent_size    
            ## get backgrounds for mixed cue condition
            # get backgrounds for location only and voice + location conditions
            bg_scene = self.dataset['bg_scene'][start : normal_ix].transpose((0, 2, 1))
            bg_scene_co_located = self.dataset['bg_scene_co_located'][normal_ix : co_located_ix].transpose((0, 2, 1))
            # first 2/3 normal, last 1/3 co-located
            background = np.concatenate((bg_scene, bg_scene_co_located), axis=0) 
        else:
            background = self.dataset['bg_scene'][start:end].transpose((0, 2, 1))

        if self.task == 'word':
            label = self.dataset[self.label_key][start:end]
        elif self.task == 'location':
            azim = self.dataset[self.label_key[0]][start:end]
            elev = self.dataset[self.label_key[1]][start:end]
            label = self.azim_elev_to_label(azim, elev)

        else:
            word = self.dataset[self.label_key[0]][start:end]
            azim = self.dataset[self.label_key[1]][start:end]
            elev = self.dataset[self.label_key[2]][start:end]
            loc = self.azim_elev_to_label(azim, elev)
            label = np.stack((word, loc), axis=1)

        if self.clean_percentage > 0.0:
            # num_clean = int(self.clean_percentage * self.batch_size)
            # clean_idx = np.random.choice(self.batch_ixs, num_clean, replace=False)
            clean_idx = self.ix_probs.multinomial(self.n_to_sample, replacement=False)
            background[clean_idx, :, :] = 0
            if self.no_cue_on_clean:
                cue[clean_idx, :, :] = 0

        if self.run_mono:
            # Just take single channel
            cue = cue[:,0,:].reshape(self.batch_size, -1)
            target = target[:,0,:].reshape(self.batch_size, -1)
            background = background[:,0,:].reshape(self.batch_size, -1)
            
        if self.mono_sanity_check:
            # running diotic. Sum channels then copy to both channels
            cue = np.sum(cue, axis=1, keepdims=True).repeat(2, axis=1)
            target = np.sum(target, axis=1, keepdims=True).repeat(2, axis=1)
            background = np.sum(background, axis=1, keepdims=True).repeat(2, axis=1)
            
        return cue, target, background, label

    def __len__(self):
        return self.dataset_len