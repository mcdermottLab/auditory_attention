import torch
import pickle
import numpy as np
import pandas as pd 
from torch.utils.data import Dataset


class TIMIT_WSN(Dataset):
    def __init__(self, root, mode='test', n_talkers=1, transform=None, noise_only=False, demo=False):
        """
        Builds a pytorch dataset from a pandas dataframe
        Args:
            root (str): location of the pd dataframe
        """
        self.path = root
        self.coch_transform = transform[0] # has cochleagram transform after rms normalization & fg-bg combination
        self.mix_transform = transform[1] # rms normalizes and combines signals at random SNR without cochleagram
        self.dataset = pd.read_pickle(self.path)
        self.n_talkers = n_talkers 
        self.demo = demo 

        if isinstance(n_talkers, (list, tuple)):
            self.get_bg_talker_ixs = self.get_random_n_talker_ixs
        else:
            self.get_bg_talker_ixs = self.get_talker_ixs

        self.dataset_len = self.dataset.shape[0]

    def class_map(self):
        """
        Loads the mapping between the word IDX and human readable word map. 
        """
        word_and_speaker_encodings = pickle.load( open( "/om4/group/mcdermott/user/jfeather/projects/model_metamers/figure_generation_notebooks/word_and_speaker_encodings_jsinv3.pckl", "rb" )) 
        class_map = word_and_speaker_encodings['word_idx_to_word']
        return class_map

    def get_talker_ixs(self, background_ixs):
        '''Randomly choose fixed number of talkers'''
        talker_ixs = np.random.choice(background_ixs, size=self.n_talkers, replace=False)
        return talker_ixs

    def get_random_n_talker_ixs(self, background_ixs):
        '''Randomly choose number of talkers from provided upper and lower bounds.
        Add one to high as upper bound of np.random.randint is not inclusive
        '''
        n_talkers = np.random.randint(low=self.n_talkers[0], high=self.n_talkers[1]+1) 
        talker_ixs = np.random.choice(background_ixs, size=n_talkers, replace=False)
        talker_ixs = np.sort(talker_ixs)
        return talker_ixs 

    def __getitem__(self, index):
        """
        Gets components of the pd dataframe that are used for training
        Args: 
            index (int): index into the pd dataframe
        Returns:
            [signal, fg_cue, fg_target] : the training audio (signal) post preprocessing
              which may combine the foreground and background speech, the training audio cue 
              post processing, and the target word idx. 
            
        """
        foreground = self.dataset.signal[index]
        talker = self.dataset.speaker[index]
        fg_target = self.dataset.word_int[index]

        cue_ixs = np.where(self.dataset.speaker == talker)[0]
        cue_ixs = cue_ixs[cue_ixs != index] # don't include target excerpt
        cue_ix = np.random.choice(cue_ixs)
        assert self.dataset.speaker[cue_ix] == talker, "Cue selected from different talker!"
        assert cue_ix != index, "Cue excerpt cannot be the same as foreground!"
        fg_cue = self.dataset.signal[cue_ix]

        # get background talkers
        background_ixs = np.where(self.dataset.speaker != talker)[0]
        talker_ixs = self.get_bg_talker_ixs(background_ixs)
        assert index not in talker_ixs, "Background talker same as target talker!"
        background_talkers = np.stack(self.dataset.signal[talker_ixs]) # stack to get BxT sized array
        # Transforms will take in the signal and the noise source for this dataset
        if self.n_talkers != 1:
            # mix talkers at random SNRs:
            for ix, talker in enumerate(background_talkers):
                if ix == 0:
                    # [0] to select signal. mix_transform returns fg, bg pairs - here bg is none 
                    background = self.mix_transform(talker, None)[0].squeeze().numpy() 
                else:
                    # [0] to select signal. mix_transform returns fg, bg pairs - here bg is none 
                    background = self.mix_transform(talker, background)[0].squeeze().numpy() 
        else:
            background = background_talkers.squeeze()

        # get cochleagrams of target in noise and of cue 
        signal, _ = self.coch_transform(foreground, background)
        fg_cue, _ = self.coch_transform(fg_cue, None)

        if self.demo:
            return foreground, background, signal, fg_cue, fg_target
        return signal, fg_cue, fg_target
