import torch
import librosa
import pickle
from pathlib import Path


class SWCMonoTestSet(torch.utils.data.Dataset):
    def __init__(self, stim_path, cond_ix, model_sr, label_type="WSN"):
        stim_path = Path(stim_path) / f"condition_{cond_ix:02d}"
        self.walker = list(stim_path.glob("*.wav"))
        with open("human_saddler_attn_expmt_cond_map.pkl", "rb") as f:
            self.stim_cond_map = pickle.load(f)
        # get word key 
        with open("human_saddler_attn_expmt_word_key.pkl", "rb") as f:
            self.word_key = pickle.load(f)       
        self.len = len(self.walker)
        self.sr = model_sr
        self.cue_end_frame = int(2 * self.sr)
        self.mixture_start_frame = int(2.5 * self.sr)
        self.label_type = label_type
        self.class_map, self.word_2_class = self.get_class_map()

    def get_class_map(self):
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
        # get stimulus 
        excerpt_path = self.walker[index]
        stim, _ = librosa.load(excerpt_path, sr=self.sr)
        cue_signal = stim[ : self.cue_end_frame]
        mixture_signal = stim[self.mixture_start_frame : ]
        word = self.word_key[int(excerpt_path.stem)]
        word_label = self.word_2_class[word]
        return cue_signal, mixture_signal, word_label

    def __len__(self):
        return self.len
