import torch
import pickle
import numpy as np
import pandas as pd 
from torch.utils.data import Dataset
import torchaudio.transforms as T


class TIMIT_WSN(Dataset):
    def __init__(self, root, mode='test', n_talkers=1, bg_label=True, transform=None, demo=False):
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
        self.bg_label = bg_label

        if isinstance(n_talkers, (list, tuple)):
            self.get_bg_talker_ixs = self.get_random_n_talker_ixs
        else:
            self.get_bg_talker_ixs = self.get_talker_ixs

        self.dataset_len = self.dataset.shape[0]

    def class_map(self):
        """
        Loads the mapping between the word IDX and human readable word map. 
        """
        word_and_speaker_encodings = pickle.load( open( "/om2/user/imgriff/projects/Auditory-Attention/word_and_speaker_encodings_jsinv3.pckl", "rb" )) 
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
        foreground = self.dataset.signal[index].astype('float32')
        talker = self.dataset.speaker[index]
        fg_target = self.dataset.word_int[index]

        cue_ixs = np.where(self.dataset.speaker == talker)[0]
        cue_ixs = cue_ixs[cue_ixs != index] # don't include target excerpt
        cue_ix = np.random.choice(cue_ixs)
        assert self.dataset.speaker[cue_ix] == talker, "Cue selected from different talker!"
        assert cue_ix != index, "Cue excerpt cannot be the same as foreground!"
        fg_cue = self.dataset.signal[cue_ix].astype('float32')

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
        background = background.astype('float32')

        # get cochleagrams of target in noise and of cue 
        signal, _ = self.coch_transform(foreground, background)
        fg_cue, _ = self.coch_transform(fg_cue, None)

        if self.demo:
            return foreground, background, signal, fg_cue, fg_target
        if self.bg_label:
            bg_target = self.dataset.word_int[talker_ixs[0]]
            # returning fg_cue for bg_cue - is not used but needed by collate fn
            return signal, fg_cue, fg_cue, fg_target, bg_target
        return signal, fg_cue, fg_target
    
    def __len__(self):
        return self.dataset_len


class TIMIT_WSN_Prepaired(Dataset):
    def __init__(self, root, mode='test', n_talkers=None, transform=None, demo=False, clean_targets=False):
        """
        Builds a pytorch dataset from a pandas dataframe that has cues and mixtures pre-cut
        Args:
            root (str): location of the pd dataframe
        """
        self.path = root
        self.coch_transform = transform[0] # set of transforms to apply to signals 
        self.dataset = pd.read_pickle(self.path)
        self.demo = demo 

        self.dataset_len = self.dataset.shape[0]

        if clean_targets:
            self.target_signals = self.dataset.signal
        else:
            self.target_signals = self.dataset.mixture_signal


    def class_map(self):
        """
        Loads the mapping between the word IDX and human readable word map. 
        """
        word_and_speaker_encodings = pickle.load( open( "/om2/user/imgriff/projects/Auditory-Attention/word_and_speaker_encodings_jsinv3.pckl", "rb" )) 
        class_map = word_and_speaker_encodings['word_idx_to_word']
        return class_map


    def __getitem__(self, index):
        """
        Gets components of the pd dataframe that are used for training
        Cues and mixtures are prepaired, so we just need to transform
        the signals.  
        Args: 
            index (int): index into the pd dataframe
        Returns:
            [mixture, cue, target_word_int] : the training audio (signal) post preprocessing
              which may combine the foreground and background speech, the training audio cue 
              post processing, and the target word idx. 
        """
        mixture = self.target_signals[index].astype('float32') # pre-mixed target and distractor 
        cue = self.dataset.cue_signal[index].astype('float32')        # pre selected cue 
        target_word_int = self.dataset.word_int[index].astype('int')  # target word label
        # transform cue and signal to cochleagrams 
        mixture, _ = self.coch_transform(mixture, None)
        cue, _ = self.coch_transform(cue, None)

        if self.demo:
            target = self.dataset.signal[index]
            distractor = self.dataset.distractor_signal[index]
            return target, distractor, mixture, cue, target_word_int
        return mixture, cue, target_word_int
    
    def __len__(self):
        return self.dataset_len


class TIMIT_CV_Compat_Prepaired(Dataset):
    def __init__(self, root, mode='test', demo=False, return_cue=False, clean_targets=True, **kwargs):
        """
        Builds a pytorch dataset from a pandas dataframe that has cues and mixtures pre-cut
        Args:
            root (str): location of the pd dataframe
        """
        self.path = root
        self.dataset = pd.read_pickle(self.path)
        self.demo = demo 
        self.return_cue = return_cue
        self.dataset_len = self.dataset.shape[0]
        self.upsample = T.Resample(20_000, 50_000, dtype=torch.float32)

        if clean_targets:
            self.target_signals = self.dataset.signal
        else:
            self.target_signals = self.dataset.mixture_signal
        self.class_remap, self.cv_class_map = self.get_class_map()

    def get_class_map(self):
        """
        Loads the mapping between the word IDX and human readable word map. 
        """
        word_and_speaker_encodings = pickle.load( open( "/om2/user/imgriff/projects/Auditory-Attention/word_and_speaker_encodings_jsinv3.pckl", "rb" )) 
        # key is int, val is word
        wsn_class_map = word_and_speaker_encodings['word_idx_to_word']
        # key is word, val is int
        cv_class_map = pickle.load( open("/om2/user/imgriff/datasets/commonvoice_9/en/cv_word_int_label_dict.pkl", "rb" )) 
        # map wsn class int key to cv class int value 
        class_remap = {ix:(cv_class_map[word] if word in cv_class_map else 0) for ix, word in wsn_class_map.items()}
        class_map = {ix:word for word,ix in cv_class_map.items()}
        return class_remap, class_map

    def __getitem__(self, index):
        """
        Gets components of the pd dataframe that are used for testings
        Cues and mixtures are prepaired, so we just need to transform
        the signals.  
        Args: 
            index (int): index into the pd dataframe
        Returns:
            [mixture, cue, target_word_int] : the training audio (signal) post preprocessing
              which may combine the foreground and background speech, the training audio cue 
              post processing, and the target word idx. 
        """
        mixture = self.target_signals[index].astype('float32') # pre-mixed target and distractor 
        mixture = self.upsample(torch.from_numpy(mixture)).numpy()
        target_word_int = self.dataset.word_int[index].astype('int')  # target word label
        # map to cv vocab
        target_word_int = self.class_remap[target_word_int]
        if self.return_cue:
            cue = self.dataset.cue_signal[index].astype('float32')        # pre selected cue 

        if self.demo:
            target = self.dataset.signal[index]
            distractor = self.dataset.distractor_signal[index]
            return target, distractor, mixture, cue, target_word_int
        if self.return_cue:
            return mixture, cue, target_word_int
        return mixture, target_word_int
    
    def __len__(self):
        return self.dataset_len



class TIMIT_Binaural_Compat_Prepaired(Dataset):
    def __init__(self, root, mode='test', demo=False, clean_targets=False, run_mono=False, **kwargs):
        """
        Builds a pytorch dataset from a pandas dataframe that has cues and mixtures pre-cut
        Args:
            root (str): location of the pd dataframe
        """
        self.path = root
        self.dataset = pd.read_pickle(self.path)
        self.demo = demo 
        self.dataset_len = self.dataset.shape[0]
        self.upsample = T.Resample(20_000, 50_000, dtype=torch.float32)
        self.run_mono = run_mono

        if clean_targets:
            self.target_signals = self.dataset.signal
        else:
            self.target_signals = self.dataset.mixture_signal
        self.class_remap, self.cv_class_map = self.get_class_map()

    def get_class_map(self):
        """
        Loads the mapping between the word IDX and human readable word map. 
        """
        word_and_speaker_encodings = pickle.load( open( "/om2/user/imgriff/projects/Auditory-Attention/word_and_speaker_encodings_jsinv3.pckl", "rb" )) 
        # key is int, val is word
        wsn_class_map = word_and_speaker_encodings['word_idx_to_word']
        # key is word, val is int
        cv_class_map = pickle.load( open("/om2/user/imgriff/datasets/commonvoice_9/en/cv_800_word_label_to_int_dict.pkl", "rb" )) 
        # map wsn class int key to cv class int value 
        class_remap = {ix:(cv_class_map[word] if word in cv_class_map else -1) for ix, word in wsn_class_map.items()}
        class_map = {ix:word for word,ix in cv_class_map.items()}
        return class_remap, class_map

    def __getitem__(self, index):
        """
        Gets components of the pd dataframe that are used for testings
        Cues and mixtures are prepaired, so we just need to transform
        the signals.  
        Args: 
            index (int): index into the pd dataframe
        Returns:
            [cue, mixture, None, target_word_int] : the training audio (signal) post preprocessing
              which may combine the foreground and background speech, the training audio cue 
              post processing, and the target word idx. 
        """
        mixture = self.target_signals[index].astype('float32') # pre-mixed target and distractor 
        mixture = self.upsample(torch.from_numpy(mixture)).numpy()
        target_word_int = self.dataset.word_int[index].astype('int')  # target word label
        # map to cv vocab
        target_word_int = self.class_remap[target_word_int]

        cue = self.dataset.cue_signal[index].astype('float32')        # pre selected cue 
        cue = self.upsample(torch.from_numpy(cue)).numpy()

        if not self.run_mono:
            # reshape to binaural compat (2,-1)
            cue = np.repeat(cue[np.newaxis,:], 2, axis=0)
            mixture = np.repeat(mixture[np.newaxis,:], 2, axis=0)

        if self.demo:
            target = self.dataset.signal[index]
            distractor = self.dataset.distractor_signal[index]
            return target, distractor, mixture, cue, target_word_int

        return cue, mixture, None, target_word_int
    
    def __len__(self):
        return self.dataset_len