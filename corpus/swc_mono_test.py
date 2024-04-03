import torch
import librosa
import pickle
import pandas as pd 
from pathlib import Path


class SWCMonoTestSet(torch.utils.data.Dataset):
    """
    Dataset class for evaluation on pre-cut mixtures used in human diotic experiments.
    """
    def __init__(self, stim_path, cond_ix, model_sr, label_type="WSN", popham_stim=False):
        stim_path = Path(stim_path) / f"condition_{cond_ix:02d}"
        self.walker = list(stim_path.glob("*.wav"))
        if popham_stim:
            with open("/om2/user/imgriff/projects/torch_2_aud_attn/swc_popham_exmpt_2024_cond_manifest.pkl", "rb") as f:
                self.stim_cond_map = pickle.load(f)
        else:
            with open("/om2/user/imgriff/projects/Auditory-Attention/human_saddler_attn_expmt_cond_map.pkl", "rb") as f:
                self.stim_cond_map = pickle.load(f)
        # get word key - shared for saddler attn and popham expmt 
        with open("/om2/user/imgriff/projects/Auditory-Attention/human_saddler_attn_expmt_word_key.pkl", "rb") as f:
            self.word_key = pickle.load(f)       
        self.len = len(self.walker)
        self.sr = model_sr
        self.cue_end_frame = int(2 * self.sr)
        self.mixture_start_frame = int(2.5 * self.sr)
        self.label_type = label_type
        self.word_2_class = self.get_class_map()

    def get_class_map(self):
        """
        Loads the mapping between the word IDX and human readable word map. 
        """
        if self.label_type == "WSN":
            ## load WSN vocab mapping 
            word_and_speaker_encodings = pickle.load( open( "/om2/user/imgriff/projects/Auditory-Attention/word_and_speaker_encodings_jsinv3.pckl", "rb" )) 
            word_2_class = word_and_speaker_encodings['word_to_idx']
        elif self.label_type == "CV":
            word_2_class = pickle.load( open("/om2/user/imgriff/datasets/commonvoice_9/en/cv_800_word_label_to_int_dict.pkl", "rb" )) 
        return word_2_class

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
   

class SWCMonoTestSetUnfamililarLanguage(torch.utils.data.Dataset):
    def __init__(self, manifest_path, model_sr, distractor_language="english", label_type="WSN"):

        self.dataset = pd.read_pickle(manifest_path)
        self.len = self.dataset.shape[0]
        self.sr = model_sr
        self.label_type = label_type
        self.word_2_class = self.get_class_map()
        self.distractor_language = distractor_language
        if self.distractor_language == 'english':
            self.distractor_col = 'distractor_src_fn'
        elif self.distractor_language == 'dutch':
            self.distractor_col = 'nl_distractor_src_fn'
        if self.distractor_language == 'mandarin':
            self.distractor_col = 'zh_distractor_src_fn'
        self.dataset = self.dataset[["src_fn",
                                     "cue_src_fn",
                                     self.distractor_col,
                                     "word_int"]].values
        print(f"Evaluating using distractors from: {self.distractor_col}")

    def get_class_map(self):
        """
        Loads the mapping between the word IDX and human readable word map. 
        """
        if self.label_type == "WSN":
            ## load WSN vocab mapping 
            word_and_speaker_encodings = pickle.load( open( "/om2/user/imgriff/projects/Auditory-Attention/word_and_speaker_encodings_jsinv3.pckl", "rb" )) 
            word_2_class = word_and_speaker_encodings['word_to_idx']
        elif self.label_type == "CV":
            word_2_class = pickle.load( open("/om2/user/imgriff/datasets/commonvoice_9/en/cv_800_word_label_to_int_dict.pkl", "rb" )) 
        return word_2_class

    def __getitem__(self, index):
        # get stimulus 
        target_path, cue_path, dist_path, word_int = self.dataset[index]
        target, _ = librosa.load(target_path, sr=self.sr)
        cue, _ = librosa.load(cue_path, sr=self.sr)
        distractor, _ = librosa.load(dist_path, sr=self.sr)

        return cue, target, distractor, word_int

    def __len__(self):
        return self.len
