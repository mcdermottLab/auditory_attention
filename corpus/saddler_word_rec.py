# write torch dataloader for test set 


import torch
import pandas as pd 
import librosa
import pickle
from pathlib import Path

class SaddlerSWCWordRecTest(torch.utils.data.Dataset):
    """
    Dataset for the Saddler word recognition experiment using
    foreground excerpts from Spoken Wikipedia.  Backgrounds are 
    either: 
        - music from musdb
        - 8 talkerbabble from common voice
        - spectrally matched noise (SSN; match each foreground)
        - Festen and Plomp style modulated maskers
        - audioset
        - natural scenes from IEEE AASP CASA challenge
        - clean (no background)
    """
    def __init__(self, manifest_path, bg_stim_path, condition, label_type="WSN", sr=20_000):
        """
        Args:
            manifest_path (str): path to pandas manifest with fg and cue excerpts
            bg_stim_path (str): path to directory with background stimuli
            condition (str): background condition. Either "music", "babble", "stationary", "modulated", "audioset", "ieee_scenes", or "clean"
            label_type (str): Set of word class labels to use. Either "WSN" (JSIN) or "CV" common voice
            sr (int): sampling rate to load audio at
        """ 
        self.manifest = pd.read_pickle(manifest_path)
        self.condition = condition
        self.label_type = label_type
        self.sr = sr 
        self.condition_dict = {'music':"background_musdb18hq",
                       "babble":"background_cv08talkerbabble",
                       "stationary": "background_issnstationary",
                       "modulated": "background_issnfestenplomp",
                       "audioset": "background_audioset",
                       "ieee_scenes": "background_ieeeaaspcasa",
                       }
        
        if condition == "clean":
            self.bg_stim = None
            self.test_cond_dir = None

        else:
        
            self.test_cond_dir = self.condition_dict[self.condition]
            self.bg_stim = list((bg_stim_path / self.test_cond_dir).glob("*.wav"))
        self.class_map, self.word_2_class = self.class_map()

        self.dataset_len = len(self.manifest)

    def class_map(self):
        """
        Loads the mapping between the word IDX and human readable word map. 
        """
        if self.label_type == "WSN":
            ## load WSN vocab mapping 
            word_and_speaker_encodings = pickle.load( open( "/om2/user/imgriff/projects/Auditory-Attention/word_and_speaker_encodings_jsinv3.pckl", "rb" )) 
            class_map = word_and_speaker_encodings['word_idx_to_word']
        elif self.label_type == "CV":
            class_map = pickle.load( open("/om2/user/imgriff/datasets/commonvoice_9/en/cv_800_word_label_to_int_dict.pkl", "rb" )) 
        word_2_class = {v:k for k,v in class_map.items()}
        return class_map, word_2_class

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
        foreground, _ = librosa.load(self.manifest['src_fn'][index], sr=self.sr)
        cue, _ = librosa.load(self.manifest['cue_src_fn'][index], sr=self.sr)
        if self.condition == "clean":
            background = None
        else:
            background, _  = librosa.load(self.bg_stim[index], sr=self.sr)
    
        word = self.manifest['word'][index]
        word_label = self.word_2_class[word]
    
        return cue, foreground, background, word_label

    def __len__(self):
        return self.dataset_len
