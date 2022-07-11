import h5py
import torch
import glob
import sys
# sys.path.append('/om4/group/mcdermott/user/imgriff/projects/End-to-end-ASR-Pytorch')
# import src.audio_transforms as audio_transforms
import pickle
import numpy as np


class jsinV3_attn_tracking_validation(torch.utils.data.ConcatDataset):
    # Makes a dataset using pre-paired speech and audioset background sounds
    # Works with hdf5 files for the jsinv3 dataset, located at the following on openmind
    # /om4/group/mcdermott/projects/ibmHearingAid/assets/data/datasets/JSIN_v3.00/nStim_20000/2000ms/rms_0.1/noiseSNR_-10_10/stimSR_20000/reverb_none/noise_all/JSIN_all_v3/subsets'
    hdf5_glob = 'JSIN_all__run_*.h5'
    target_keys = ['signal/word_int']

    def __init__(self, root, train=True, download=False, transform=None, demo=False):
        """
        Builds the pytorch hdf5 combined dataset from the files found in the 
        specified root directory. 
        """
        del download

        if train:
            self.all_hdf5_files = glob.glob(root + '/train_*/' + self.hdf5_glob)
        else:
            self.all_hdf5_files = glob.glob(root + '/valid_*/' + self.hdf5_glob)[0:1] # Just get one set of them

        self.all_hdf5_datasets = [H5Dataset(h5_file, transform, self.target_keys, demo) for h5_file in self.all_hdf5_files]

        super().__init__(self.all_hdf5_datasets)

    def class_map(self):
        """
        Loads the mapping between the word IDX and human readable word map. 
        """
        word_and_speaker_encodings = pickle.load( open( "/om4/group/mcdermott/user/jfeather/projects/model_metamers/figure_generation_notebooks/word_and_speaker_encodings_jsinv3.pckl", "rb" )) 
        class_map = word_and_speaker_encodings['word_idx_to_word']
        return class_map


class H5Dataset(torch.utils.data.Dataset):
    def __init__(self, path, transform, target_keys, demo):
        """
        Builds a pytorch hdf5 dataset
        Args:
            path (str): location of the hdf5 dataset
        """
        self.file_path = path
        self.dataset = None
        self.transform = transform
        self.target_keys = target_keys
        self.demo = demo
        # TODO: implement chunking the hdf5 file so that we can shuffle the data
        # TODO: implement shuffling the audioset and the speech separately
        # self.chunk_size = hdf5_chunk_size
        with h5py.File(self.file_path, 'r', swmr=True) as file:
            self.dataset_len = len(file['sources']['signal']['signal'])

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

        # get background
        background_ixs = np.where(speakers[:] != talker)[0]
        background_ix = np.random.choice(background_ixs)
        assert speakers[background_ix] != talker, "Background talker same as target talker!"
        background = signals[background_ix]
        background_talker = speakers[background_ix]
        
        # get background cue
        cue_ixs = np.where(speakers[:] == background_talker)[0]
        cue_ixs = cue_ixs[cue_ixs != background_ix] # don't include target excerpt
        cue_ix = np.random.choice(cue_ixs)
        assert speakers[cue_ix] == background_talker, "Cue selected from different talker!"
        assert cue_ix != background_ix, "Cue excerpt cannot be the same as background!"
        bg_cue = signals[cue_ix]  

        # Transforms will take in the signal and the noise source for this dataset
        # If no transform, just return the speech with no background
        if self.transform is not None:
            signal, _ = self.transform(foreground, background)
            fg_cue, _ = self.transform(fg_cue, None)
            bg_cue, _ = self.transform(bg_cue, None)
            
        if len(self.target_keys) == 1:
            target_paths = self.target_keys[0].split('/')
            fg_target = self.dataset['sources'][target_paths[0]][target_paths[1]][index]
            bg_target = self.dataset['sources'][target_paths[0]][target_paths[1]][background_ix]
            if self.target_keys[0] == 'noise/labels_binary_via_int':
                fg_target = fg_target.astype(np.float32)
                bg_target = bg_target.astype(np.float32)

        # If there are multiple keys, our target has them explicitly listed
        else:
            target = {}
            for target_key in self.target_keys:
                target_paths = target_key.split('/')
                fg_target[target_key] = self.dataset['sources'][target_paths[0]][target_paths[1]][index]
                bg_target[target_key] = self.dataset['sources'][target_paths[0]][target_paths[1]][background_ix]
                if target_key == 'noise/labels_binary_via_int':
                    fg_target[target_key] = fg_target[target_key].astype(np.float32)
                    bg_target[target_key] = bg_target[target_key].astype(np.float32)
        if self.demo:
            return foreground, background, signal, fg_cue, bg_cue, fg_target, bg_target
        return signal, fg_cue, bg_cue, fg_target, bg_target

    def __len__(self):
        return self.dataset_len
