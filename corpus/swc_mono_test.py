import torch
import librosa
import pickle
import pandas as pd 
from pathlib import Path
import h5py
import sys 
sys.path.append('/om2/user/imgriff/datasets/spatial_audio_pipeline/spatial_audio_util/')
import util_audio 

class SWCMonoTestSet(torch.utils.data.Dataset):
    """
    Dataset class for evaluation on pre-cut mixtures used in human diotic experiments.
    """
    def __init__(self, stim_path, cond_ix, model_sr, label_type="WSN", popham_stim=False, unfamiliar_distractor=False, stim_cond_map=None):
        stim_path = Path(stim_path) / f"condition_{cond_ix:02d}"
        self.walker = list(stim_path.glob("*.wav"))
        if not stim_cond_map:
            if popham_stim:
                with open("/om2/user/imgriff/projects/torch_2_aud_attn/swc_popham_exmpt_2024_cond_manifest.pkl", "rb") as f:
                    self.stim_cond_map = pickle.load(f)
            elif unfamiliar_distractor:
                with open("/om/user/imgriff/datasets/human_distractor_language_2024/human_distractor_language_cond_map.pkl", "rb") as f:
                    self.stim_cond_map = pickle.load(f)
            else:
                with open("/om2/user/imgriff/projects/Auditory-Attention/human_saddler_attn_expmt_cond_map.pkl", "rb") as f:
                    self.stim_cond_map = pickle.load(f)
        else:
            with open(stim_cond_map, "rb") as f:
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
            word_2_class = pickle.load( open("./cv_800_word_label_to_int_dict.pkl", "rb" )) 
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
   

class SWCMonoTestSet2024(torch.utils.data.Dataset):
    """
    Dataset class for evaluation on pre-cut mixtures used in human diotic experiments.
    """
    def __init__(self, stim_path, cond_ix, model_sr, label_type="CV", stim_cond_map=None, **kwargs):
        stim_path = Path(stim_path) / f"condition_{cond_ix:02d}"
        # sorting makes analysis simpler - puts model test stim in order of manifest 
        self.walker = sorted(list(stim_path.glob("*.wav"))) 
        if not stim_cond_map:
            stim_cond_map = "/om/user/imgriff/datasets/human_word_rec_SWC_2024/human_attn_expmt_cond_map.pkl"
        with open(stim_cond_map, "rb") as f:
            self.stim_cond_map = pickle.load(f)
        # get word key for experiment - maps filename ints to words
        with open("/om/user/imgriff/datasets/human_word_rec_SWC_2024/human_attn_expmt_word_key.pkl", "rb") as f:
            self.word_key = pickle.load(f)       
        self.len = len(self.walker)
        self.sr = model_sr
        self.cue_end_frame = int(2 * self.sr)
        self.mixture_start_frame = int(2.5 * self.sr)
        self.label_type = label_type
        self.word_2_class = self.get_class_map()
        self.class_2_word = {v: k for k, v in self.word_2_class.items()}
        self.mono = False if 'azim' in str(stim_path) else True 
        print(f"Using mono: {self.mono}")

    def get_class_map(self):
        """
        Loads the mapping between the word IDX and human readable word map. 
        """
        word_2_class = pickle.load( open("/om2/user/imgriff/datasets/commonvoice_9/en/cv_800_word_label_to_int_dict.pkl", "rb" )) 
        return word_2_class

    def __getitem__(self, index):
        # get stimulus 
        excerpt_path = self.walker[index]
        stim_tag = excerpt_path.stem
        stim, _ = librosa.load(excerpt_path, sr=self.sr, mono=self.mono)
        
        cue_signal = stim[..., : self.cue_end_frame]
        mixture_signal = stim[..., self.mixture_start_frame : ]
        # get word label - stem is of form "<sex_str>_<word_int>"
        word_ix = int(excerpt_path.stem.split("_")[-1])
        word = self.word_key[word_ix]
        word_label = self.word_2_class[word]
        return cue_signal, mixture_signal, word_label, stim_tag

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


class SWCMonoTestSetH5Dataset(torch.utils.data.Dataset):
    def __init__(self, h5_path, eval_distractor_cond, model_sr, label_type="CV", for_act_analysis=False):

        self.h5_path = h5_path
        self.dataset = None
        # get distractor key 
        self.conf_key = None 
        if eval_distractor_cond == 'one_distractor':
            self.dist_key = 'one_dist_signal'
        elif eval_distractor_cond == 'four_distractor':
            self.dist_key = 'four_dist_signal'
        elif eval_distractor_cond == 'stationary':
            self.dist_key = 'ssn_dist_signal'
        elif eval_distractor_cond == 'modulated_distractor':
            self.dist_key = 'modulated_dist_signal'
        elif eval_distractor_cond == 'music':
            self.dist_key = 'music_dist_signal'
        elif eval_distractor_cond == 'babble':
            self.dist_key = 'babble_dist_signal'
        elif eval_distractor_cond == 'natural_scene':
            self.dist_key = 'nat_dist_signal'
        elif eval_distractor_cond ==  "1-talker-english-same":
            self.dist_key = '1-talker-english-same'
            self.conf_key = 'same_dist_word_int'
        elif eval_distractor_cond ==  "1-talker-english-different":
            self.dist_key = '1-talker-english-different'
            self.conf_key = 'diff_dist_word_int'
        elif eval_distractor_cond ==  "1-talker-mandarin-same":
            self.dist_key = '1-talker-mandarin-same'
        elif eval_distractor_cond ==  "1-talker-mandarin-different":
            self.dist_key = '1-talker-mandarin-different'
        elif eval_distractor_cond ==  "1-talker-dutch-same":
            self.dist_key = '1-talker-dutch-same'
        elif eval_distractor_cond ==  "1-talker-dutch-different":
            self.dist_key = '1-talker-dutch-different'
        elif eval_distractor_cond ==  "2-talker":
            self.dist_key = '2-talker'
        elif eval_distractor_cond ==  "4-talker":
            self.dist_key = '4-talker'
        elif eval_distractor_cond == 'clean':
            self.dist_key = None
        else:
            raise ValueError(f"Unknown eval_distractor_cond: {eval_distractor_cond}")

        if self.conf_key == None and "2024" in str(h5_path):
            self.conf_key = 'same_dist_word_int'
        elif self.conf_key == None:
            self.conf_key = 'confusion_int_label'

        self.sr = model_sr
        self.label_type = label_type
        self.for_act_analysis = for_act_analysis

        with h5py.File(self.h5_path, 'r', swmr=True) as file:
            self.dataset_len = len(file['target_signal'])

        
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
        if self.dataset is None:
            self.dataset = h5py.File(self.h5_path, 'r', swmr=True)
        cue = self.dataset['cue_signal'][index]
        target = self.dataset['target_signal'][index]
        word_int = self.dataset['word_int_label'][index]
        if self.dist_key:
            distractor = self.dataset[self.dist_key][index]
            dist_word_int = self.dataset[self.conf_key][index]
        else:
            distractor = None
            dist_word_int = None
        if self.for_act_analysis:
            cue_f0 = self.dataset['cue_f0'][index]
            target_f0 = self.dataset['target_f0'][index]
            dist_f0 = self.dataset['one_dist_f0'][index]
            return cue, target, distractor, word_int, dist_word_int, cue_f0, target_f0, dist_f0
        return cue, target, distractor, word_int, dist_word_int

    def __len__(self):
        return self.dataset_len


class SWCMonoTestSetH5DatasetForUnitTuning(torch.utils.data.Dataset):
    def __init__(self, h5_path, model_sr):

        self.h5_path = h5_path
        self.dataset = None
        self.sr = model_sr

        with h5py.File(self.h5_path, 'r', swmr=True) as file:
            self.dataset_len = len(file['target_signal'])

    def get_class_map(self):
        """
        Loads the mapping between the word IDX and human readable word map. 
        """
        word_2_class = pickle.load( open("/om2/user/imgriff/datasets/commonvoice_9/en/cv_800_word_label_to_int_dict.pkl", "rb" )) 
        return word_2_class

    def __getitem__(self, index):
        # get stimulus 
        if self.dataset is None:
            self.dataset = h5py.File(self.h5_path, 'r', swmr=True)

        cue = self.dataset['cue_signal'][index]
        target = self.dataset['target_signal'][index]
        word_int = self.dataset['word_int_label'][index]

        # get distractor types 
        same_sex_dist = self.dataset['1-talker-english-same'][index]
        diff_sex_dist = self.dataset['1-talker-english-different'][index]
        nat_scene_dist = self.dataset['nat_dist_signal'][index]

        target_f0 = util_audio.get_avg_f0(target, self.sr, fmin=80, fmax=300)
        same_dist_f0 = util_audio.get_avg_f0(same_sex_dist, self.sr, fmin=80, fmax=300)
        diff_dist_f0 = util_audio.get_avg_f0(diff_sex_dist, self.sr, fmin=80, fmax=300)

        return cue, target, same_sex_dist, diff_sex_dist, nat_scene_dist, word_int, target_f0, same_dist_f0, diff_dist_f0

    def __len__(self):
        return self.dataset_len
