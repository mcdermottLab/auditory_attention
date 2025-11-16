import torch
import pickle
import numpy as np
import pandas as pd 
from torch.utils.data import Dataset
import librosa 
import sys 
sys.path.append('/om2/user/imgriff/datasets/spatial_audio_pipeline/spatial_audio_util/')
import util_audio 


def sample_df(df, group, cond1, cond2, n):
	df_1 = df[df[f'{group}'] == cond1]
	df_2 = df[df[f'{group}'] == cond2]
	df_1_sample = df_1.sample(n=n)
	df_2_sample = df_2[~df_2.word.isin(df_1_sample.word)].sample(n=n)
    # keep original ixs to track metadata in analysis scripts 
	df_1_sample = df_1_sample.reset_index()
	df_1_sample.rename(columns={'index':'full_df_index'}, inplace=True)
	df_2_sample = df_2_sample.reset_index()
	df_2_sample.rename(columns={'index':'full_df_index'}, inplace=True)
	return pd.concat([df_1_sample, df_2_sample], axis=0, ignore_index=True)

def get_subset_df(df):
	n_to_samp = df.word.nunique() // 4
	female_df = sample_df(df[df.gender == 'female'], 'sex_cond', 'same', 'different',n_to_samp)
	male_df = sample_df(df[(df.gender == 'male') & (~df.word.isin(female_df.word))], 'sex_cond', 'same', 'different', n_to_samp)
	return pd.concat([female_df, male_df], axis=0, ignore_index=True)

def get_window_bounds(clip_dur_in_s, clip_start_in_s, clip_end_in_s, total_dur=3):
	offset = (total_dur  - clip_dur_in_s) / 2
	return clip_start_in_s - offset, clip_end_in_s + offset

def pad_or_trim_to_len(x, n, mode='both', kwargs_pad={}):
    """
    Increases or decreases the length of a one-dimensional signal
    by either padding or triming the array. If the difference
    between `len(x)` and `n` is odd, this function will default to
    adding/removing the extra sample at the end of the signal.
    
    Args
    ----
    x (np.ndarray): one-dimensional input signal
    n (int): length of output signal
    mode (str): specify which end of signal to modify
        (default behavior is to symmetrically modify both ends)
    kwargs_pad (dict): keyword arguments for np.pad function
    
    Returns
    -------
    x_out (np.ndarray): one-dimensional signal with length `n`
    """
    assert len(np.array(x).shape) == 1, "input must be 1D array"
    assert mode.lower() in ['both', 'start', 'end']
    n_diff = np.abs(len(x) - n)
    if len(x) > n:
        if mode.lower() == 'end':
            x_out = x[:n]
        elif mode.lower() == 'start':
            x_out = x[-n:]
        else:
            x_out = x[int(np.floor(n_diff / 2)):-int(np.ceil(n_diff / 2))]
    elif len(x) < n:
        if mode.lower() == 'end':
            pad_width = [0, n_diff]
        elif mode.lower() == 'start':
            pad_width = [n_diff, 0]
        else:
            pad_width = [int(np.floor(n_diff / 2)), int(np.ceil(n_diff / 2))]
        kwargs = {'mode': 'constant'}
        kwargs.update(kwargs_pad)
        x_out = np.pad(x, pad_width, **kwargs)
    else:
        x_out = x
    assert len(x_out) == n
    return x_out


class SWCHumanExperimentStimDataset(Dataset):
    def __init__(self, path, run_all_stim=False, total_dur=3.0, sr=44_100, ssn_distractor=False, **kwargs):
        # write docstring for class
        """
        A dataset class for the SWC human experiment stimuli.
        Args:
            path (str): path to the pd dataframe containing the stimuli
            run_all_stim (bool): if True, all stimuli are used
            total_dur (int): total duration of the stimuli
            sr (int): sample rate of the stimuli
        """
        self.path = path
        self.total_dur = total_dur
        self.dataset = pd.read_pickle(self.path)
        self.sr = sr 
        self.ssn_distractor = ssn_distractor

        if not run_all_stim:
            # run gender balanced subset of using each word only once 
            self.dataset = get_subset_df(self.dataset)
        else:
             # add full_df_index to dataset 
            self.dataset = self.dataset.reset_index()
            self.dataset.rename(columns={'index':'full_df_index'}, inplace=True)

        self.dataset_len = self.dataset.shape[0]
        self.class_map = self.get_class_map()
      
    def get_class_map(self):
        """
        Loads the mapping between the word IDX and human readable word map. 
        """
        return pickle.load( open("./cv_800_word_label_to_int_dict.pkl", "rb" )) 
    

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
        row = self.dataset.iloc[index]

        # track ix for metadata 
        meta_ix = row['full_df_index']
        
        # get target 
        src_fn = row['src_fn']
        clip_start_in_s = row['clip_start_in_s']
        clip_end_in_s = row['clip_end_in_s']
        clip_dur_in_s = row['clip_dur_in_s']
        word = row['word']
        onset, offset = get_window_bounds(clip_dur_in_s, clip_start_in_s, clip_end_in_s, self.total_dur)
        target_wav, _ = librosa.load(src_fn, sr=self.sr, offset=onset, duration=self.total_dur, dtype=np.float32)
        target_wav = pad_or_trim_to_len(target_wav, int(self.sr * self.total_dur))

        cue_src_fn = row['cue_src_fn']
        cue_clip_start_in_s = row['cue_clip_start_in_s']
        cue_clip_end_in_s = row['cue_clip_end_in_s']
        cue_clip_dur_in_s = row['cue_clip_dur_in_s']
        cue_onset, cue_offset = get_window_bounds(cue_clip_dur_in_s, cue_clip_start_in_s, cue_clip_end_in_s, self.total_dur)
        cue_wav, _ = librosa.load(cue_src_fn, sr=self.sr, offset=cue_onset, duration=self.total_dur, dtype=np.float32)
        cue_wav = pad_or_trim_to_len(cue_wav, int(self.sr * self.total_dur))

        dist_src_fn = row['distractor_src_fn'][0]
        dist_clip_start_in_s = row['distractor_clip_start_in_s'][0]
        dist_clip_end_in_s = row['distractor_clip_end_in_s'][0]
        dist_clip_dur_in_s = row['distractor_clip_dur_in_s'][0]
        dist_word_1 = row['distractor_word'][0]

        dist_onset, dist_offset = get_window_bounds(dist_clip_dur_in_s, dist_clip_start_in_s, dist_clip_end_in_s, self.total_dur)
        distractor_1_wav, _ = librosa.load(dist_src_fn, sr=self.sr, offset=dist_onset, duration=self.total_dur, dtype=np.float32)
        distractor_1_wav = pad_or_trim_to_len(distractor_1_wav, int(self.sr * self.total_dur))


        dist_src_fn = row['distractor_src_fn'][1]
        dist_clip_start_in_s = row['distractor_clip_start_in_s'][1]
        dist_clip_end_in_s = row['distractor_clip_end_in_s'][1]
        dist_clip_dur_in_s = row['distractor_clip_dur_in_s'][1]
        dist_word_2 = row['distractor_word'][1]
        dist_onset, dist_offset = get_window_bounds(dist_clip_dur_in_s, dist_clip_start_in_s, dist_clip_end_in_s, self.total_dur)
        distractor_2_wav, _ = librosa.load(dist_src_fn, sr=self.sr, offset=dist_onset, duration=self.total_dur, dtype=np.float32)
        distractor_2_wav = pad_or_trim_to_len(distractor_2_wav, int(self.sr * self.total_dur))

        # map to cv vocab
        target_word_int = self.class_map[word]
        dist_word_int_1 = self.class_map[dist_word_1]
        dist_word_int_2 = self.class_map[dist_word_2]

        if self.ssn_distractor:
            # get distractor: spectrally matched noise
            distractor_1_wav = util_audio.spectrally_matched_noise(target_wav, self.sr).astype('float32')
            return cue_wav, target_wav, distractor_1_wav, _, target_word_int, dist_word_int_1, _, meta_ix


        return cue_wav, target_wav, distractor_1_wav, distractor_2_wav, target_word_int, dist_word_int_1, dist_word_int_2, meta_ix
    
    def __len__(self):
        return self.dataset_len