from re import L
import h5py
import torch
import glob
import sys
# sys.path.append('/om4/group/mcdermott/user/imgriff/projects/End-to-end-ASR-Pytorch')
# import src.audio_transforms as audio_transforms
import pickle
import numpy as np


class jsinV3_attn_cue_multi_source(torch.utils.data.ConcatDataset):
    # Makes a dataset using pre-paired speech and audioset background sounds
    # Works with hdf5 files for the jsinv3 dataset, located at the following on openmind
    # /om4/group/mcdermott/projects/ibmHearingAid/assets/data/datasets/JSIN_v3.00/nStim_20000/2000ms/rms_0.1/noiseSNR_-10_10/stimSR_20000/reverb_none/noise_all/JSIN_all_v3/subsets'
    hdf5_glob = 'JSIN_all__run_*.h5'
    target_keys = ['signal/word_int']

    def __init__(self, root, mode='train', download=False, transform=None,
                 n_talkers=1, noise_only=None, with_audioset=False, demo=False):
        """
        Builds the pytorch hdf5 combined dataset from the files found in the 
        specified root directory. 
        """
        del download
        del noise_only

        if mode == 'train':
            self.all_hdf5_files = glob.glob(root + '/train_*/' + self.hdf5_glob)
        elif mode == 'val':
            self.all_hdf5_files = glob.glob(root + '/valid_*/' + self.hdf5_glob)[0:1] # Just get one set of them
        elif mode == 'test':
            self.all_hdf5_files = glob.glob(root + '/valid_*/' + self.hdf5_glob)[1:] # Use the others 

        self.all_hdf5_datasets = [H5Dataset(h5_file, transform, self.target_keys, n_talkers, with_audioset, demo) for h5_file in self.all_hdf5_files]

        super().__init__(self.all_hdf5_datasets)

    def class_map(self):
        """
        Loads the mapping between the word IDX and human readable word map. 
        """
        word_and_speaker_encodings = pickle.load( open( "/om4/group/mcdermott/user/jfeather/projects/model_metamers/figure_generation_notebooks/word_and_speaker_encodings_jsinv3.pckl", "rb" )) 
        class_map = word_and_speaker_encodings['word_idx_to_word']
        return class_map


class H5Dataset(torch.utils.data.Dataset):
    def __init__(self, path, transform, target_keys, n_talkers, with_audioset, demo):
        """
        Builds a pytorch hdf5 dataset
        Args:
            path (str): location of the hdf5 dataset
        """
        self.file_path = path
        self.dataset = None
        # has cochleagram transform after rms normalization & fg-bg combination - from audio attention transforms 
        self.coch_transform = transform[0] 
        # rms normalizes - from audio transforms
        self.mix_transform = transform[1] 
        self.target_keys = target_keys
        self.n_talkers = n_talkers

        if isinstance(n_talkers, (list, tuple)):
            self.get_bg_talker_ixs = self.get_random_n_talker_ixs
        else:
            self.get_bg_talker_ixs = self.get_talker_ixs

        self.with_audioset = with_audioset

        # If using audioset noise, return the noise example, if not return none 
        if with_audioset:
            self.get_noise = self.get_noise_signal
        else:
            self.get_noise = lambda x: None 
            
        self.demo = demo
        # TODO: implement chunking the hdf5 file so that we can shuffle the data
        # TODO: implement shuffling the audioset and the speech separately
        # self.chunk_size = hdf5_chunk_size
        with h5py.File(self.file_path, 'r', swmr=True) as file:
            self.dataset_len = len(file['sources']['signal']['signal'])

    def get_talker_ixs(self, background_ixs):
        '''Randomly choose fixed number of talkers'''
        talker_ixs = np.random.choice(background_ixs, size=self.n_talkers, replace=False)
        if self.n_talkers > 1:
            talker_ixs = np.sort(talker_ixs)
        return talker_ixs

    def get_random_n_talker_ixs(self, background_ixs):
        '''Randomly choose number of talkers from provided upper and lower bounds.
        Add one to high as upper bound of np.random.randint is not inclusive
        '''
        n_talkers = np.random.randint(low=self.n_talkers[0], high=self.n_talkers[1]+1) 
        talker_ixs = np.random.choice(background_ixs, size=n_talkers, replace=False)
        talker_ixs = np.sort(talker_ixs)
        return talker_ixs 

    def get_noise_signal(self, index):
        return self.dataset['sources']['noise']['signal'][index]
        
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
            self.dataset = h5py.File(self.file_path, 'r', swmr=True)# ["ndarray_data"]["signal"]
            # print(self.dataset['sources']['signal'].keys())
        # TODO: this should apply the transform to the signal automatically, and then return. 
      
        # set handles for access         
        speakers = self.dataset['sources']['signal']['speaker_int']
        signals = self.dataset['sources']['signal']['signal'] 

        # get  noise
        noise = self.get_noise(index)

        # Get foreground
        foreground = signals[index]
        talker = speakers[index]

        # get foreground cue 
        cue_ixs = np.where(speakers[:] == talker)[0]
        cue_ixs = cue_ixs[cue_ixs != index] # don't include target excerpt
        cue_ix = np.random.choice(cue_ixs)
        assert speakers[cue_ix] == talker, "Cue selected from different talker!"
        assert cue_ix != index, "Cue excerpt cannot be the same as foreground!"
        fg_cue = signals[cue_ix]

        # get background talkers
        background_ixs = np.where(speakers[:] != talker)[0]
        talker_ixs = self.get_bg_talker_ixs(background_ixs)
        assert index not in talker_ixs, "Background talker same as target talker!"
        assert (np.diff(talker_ixs) > 0).all(), "Background indices not ascending"
        background_talkers = signals[talker_ixs, :]
        # Transforms will take in the signal and the noise source for this dataset
        # # mix talkers at random SNRs:
        # for ix, talker in enumerate(background_talkers):
        #     if ix == 0:
        #         background_talkers = self.mix_transform(talker, None)[0].squeeze().numpy() # [0] to select signal. mix_transform returns fg, bg pairs - here bg is none 
        #     else:
        #         background_talkers = self.mix_transform(talker, background_talkers)[0].squeeze().numpy() # [0] to select signal. mix_transform returns fg, bg pairs - here bg is none 
        background = [self.mix_transform(bg, None)[0].squeeze().numpy() for bg in background_talkers]
        background = np.sum(background, axis=0)
        # mix audioset and talkers 
        background = self.mix_transform(background_talkers, noise)[0].squeeze().numpy() # [0] to select signal. mix_transform returns fg, bg pairs - here bg is none 
        # get cochleagrams of target in noise and of cue 
        fg_cue, signal, _ = self.coch_transform(fg_cue, foreground, background)
            
        if len(self.target_keys) == 1:
            target_paths = self.target_keys[0].split('/')
            fg_target = self.dataset['sources'][target_paths[0]][target_paths[1]][index]
            if self.target_keys[0] == 'noise/labels_binary_via_int':
                fg_target = fg_target.astype(np.float32)

        # If there are multiple keys, our target has them explicitly listed
        else:
            fg_target = {}
            for target_key in self.target_keys:
                target_paths = target_key.split('/')
                fg_target[target_key] = self.dataset['sources'][target_paths[0]][target_paths[1]][index]
                if target_key == 'noise/labels_binary_via_int':
                    fg_target[target_key] = fg_target[target_key].astype(np.float32)
        if self.demo:
            return foreground, background, signal, fg_cue, fg_target
        return signal, fg_cue, fg_target

    def __len__(self):
        return self.dataset_len
