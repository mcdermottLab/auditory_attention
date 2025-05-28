import torch
import pickle
import numpy as np 
from pathlib import Path
import h5py

class SpeechAndTextureTestSet(torch.utils.data.Dataset):
    def __init__(self, file_path, label_type="CV", separated_signals=False, symmetric_distractor=False, **kwargs):
        self.file_path  = Path(file_path) 
        self.dataset = None
        self.label_type = label_type
        self.word_2_class = self.get_class_map()
        self.class_2_word = {v: k for k, v in self.word_2_class.items()}
        self.separated_signals = separated_signals
        self.symmetric_distractor = symmetric_distractor

        with h5py.File(self.file_path, 'r', swmr=True) as file:
            self.dataset_len = len(file['cue_signal'])

    def get_class_map(self):
        """
        Loads the mapping between the word IDX and human readable word map. 
        """
        if self.label_type == "WSN":
            ## load WSN vocab mapping 
            word_and_speaker_encodings = pickle.load( open( "/om2/user/imgriff/projects/Auditory-Attention/word_and_speaker_encodings_jsinv3.pckl", "rb" )) 
            word_2_class = word_and_speaker_encodings['word_to_idx']
        elif self.label_type == "CV":
            word_2_class = pickle.load( open("./cv_800_word_label_to_int_dict.pkl", "rb" )) 
        return word_2_class

    def __getitem__(self, index):
        if self.dataset is None:
            self.dataset = h5py.File(self.file_path, 'r', swmr=True)
        cue_signal = self.dataset['cue_signal'][index]
        word_label = self.dataset['label_word_int'][index]
        texture = self.dataset['index_texture'][index]
        if self.separated_signals:
            target = self.dataset['target'][index]
            distractor = self.dataset['distractor'][index]
            if self.symmetric_distractor:
                distractor_2_ixs = np.where(self.dataset['index_texture'][:] == texture)[0]
                # get choice that is different than distractor
                distractor_2_ix = np.random.choice(distractor_2_ixs[distractor_2_ixs != index], 1)
                distractor_2 = self.dataset['distractor'][distractor_2_ix]
                return cue_signal, target, distractor, distractor_2, word_label, texture, index
            else:
                return cue_signal, target, distractor, word_label, texture
        else:
            mixture_signal = self.dataset['signal'][index]
            return cue_signal, mixture_signal, word_label, texture
        

    def __len__(self):
        return self.dataset_len

