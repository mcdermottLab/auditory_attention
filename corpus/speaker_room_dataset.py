import h5py
import librosa
import numpy as np
import pandas as pd
import pickle
import soundfile as sf
import soxr
import torch
import sys 

sys.path.append('/om2/user/imgriff/datasets/spatial_audio_pipeline/spatial_audio_util/')
import util_audio 

from pathlib import Path

class SpeakerRoomDataset(torch.utils.data.Dataset):
    """
    Dataset for the Griffith and Hess binaural model test using
    foreground excerpts from Spoken Wikipedia.  Backgrounds are 
    either also from SWC.
    """
    def __init__(self, manifest_path, excerpt_path, cue_type, sr=20_000, symmetric_distractor_test=False, modulated_ssn_distractors=False, return_stim_ixs=False):
        """
        Args:
            manifest_path (str): path to pandas manifest with trials defined
            excerpt_path (str): path to pandas excerpt manifest pointing to audio clips
            cue_type (str): type of cue to use, either "voice", "location", or "both"
            sr (int): sampling rate to load audio at
        """ 
        self.manifest = pd.read_pickle(manifest_path)
        self.excerpts = pd.read_pickle(excerpt_path)
        self.cue_type = cue_type
        self.sr = sr 
        self.symmetric_distractor_test = symmetric_distractor_test
        self.modulated_ssn_distractors = modulated_ssn_distractors
        self.return_stim_ixs = return_stim_ixs

        self.class_map, self.word_2_class = self.class_map()

        self.dataset_len = len(self.manifest)

    def class_map(self):
        """
        Loads the mapping between the word IDX and human readable word map. 
        """
        word_2_class = pickle.load( open("./cv_800_word_label_to_int_dict.pkl", "rb" )) 
        class_map = {v:k for k,v in word_2_class.items()}
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
        src_ix = self.manifest['src_ix'][index]
        cue_ix = self.manifest['cue_src_ix'][index]
        bg_ix = self.manifest['bg_src_ix'][index]
        # pick bg somehow
        if self.cue_type != 'location':
            cue = self.get_excerpt(self.excerpts.iloc[cue_ix])
        else:
            cue = self.whitenoise(3)
        src = self.get_excerpt(self.excerpts.iloc[src_ix])
        bg = self.get_excerpt(self.excerpts.iloc[bg_ix])

        word = self.manifest['word'][index]
        word_label = self.word_2_class[word]
        dist_word = self.manifest['bg_word'][index]
        dist_word_label = self.word_2_class[dist_word]

        if not self.symmetric_distractor_test:
            if self.return_stim_ixs:
                return cue, src, bg, word_label, dist_word_label, index
            return cue, src, bg, word_label, dist_word_label

        elif self.symmetric_distractor_test:
            bg_man_ix2 = np.random.choice(self.manifest.index)
            bg_src_ix2 = self.manifest['bg_src_ix'][bg_man_ix2]
            bg_2 = self.get_excerpt(self.excerpts.iloc[bg_src_ix2])
            dist_word2 = self.manifest['bg_word'][bg_man_ix2]
            dist_word_label2 = self.word_2_class[dist_word2]
            if self.modulated_ssn_distractors:
                # get modulated speech-shaped noise for distractor 1 and 2 
                bg = bg.numpy()
                bg_2 = bg_2.numpy()
                bg_noise = util_audio.spectrally_matched_noise(bg, self.sr)
                bg = util_audio.festen_plomp_fluctuating_noise(bg, bg_noise, sr=self.sr, two_band_cutoff=None)
                bg_noise2 = util_audio.spectrally_matched_noise(bg_2, self.sr)
                bg_2 = util_audio.festen_plomp_fluctuating_noise(bg_2, bg_noise2, sr=self.sr, two_band_cutoff=None)
                bg = torch.from_numpy(bg).float()
                bg_2 = torch.from_numpy(bg_2).float()
            if self.return_stim_ixs:
                return cue, src, bg, word_label, dist_word_label, index
            return cue, src, bg, bg_2, word_label, dist_word_label, dist_word_label2


    def __len__(self):
        return self.dataset_len

    def whitenoise(self, time, dtype=torch.float32, rms=65):
        noise = torch.rand(int(time * self.sr), dtype=dtype)
        #! Should we rms norm?
        # noise = _rms_norm(noise, rms)
        return noise

    def pad_or_trim_to_len(self, x, n, mode='both', kwargs_pad={}):
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

    def get_excerpt(self, dfi, dur=3.0, pad_with_context=True, jitter_fraction=0):
        """
        This function loads an audio file and excerpts a clip with the specified
        duration. Target durations that exceed clip boundaries are handled with
        zero-padding (applied to all signals but sliced away when not needed).
        This function also handles resampling (via soxr) and re-scaling.
        """
        jitter_in_s = 0
        jitter_via_zero_padding = True
        sr = self.sr
        if dfi.clip_dur_in_s > dur:
            # Take a random segment if clip duration is longer than excerpt
            clip_start_in_s = np.random.uniform(
                low=dfi.clip_start_in_s,
                high=dfi.clip_start_in_s + dfi.clip_dur_in_s - dur,
                size=None)
            clip_end_in_s = clip_start_in_s + dur
            jitter_via_zero_padding = False
        else:
            # Temporally jitter clip by extending either start or end time
            jitter_in_s = np.random.uniform(
                low=-dfi.clip_dur_in_s * jitter_fraction,
                high=dfi.clip_dur_in_s * jitter_fraction,
                size=None)
            if pad_with_context:
                # If using context, adjust clip start and end times to account for jitter and context
                if jitter_in_s > 0:
                    clip_start_in_s = dfi.clip_start_in_s - (2 * np.abs(jitter_in_s))
                    clip_end_in_s = dfi.clip_end_in_s
                else:
                    clip_start_in_s = dfi.clip_start_in_s
                    clip_end_in_s = dfi.clip_end_in_s + (2 * np.abs(jitter_in_s))
                clip_dur_in_s = clip_end_in_s - clip_start_in_s
                jitter_via_zero_padding = False
                context_pad_in_s = (dur - clip_dur_in_s) / 2
            else:
                clip_start_in_s = dfi.clip_start_in_s
                clip_end_in_s = dfi.clip_end_in_s
                context_pad_in_s = 0
            clip_start_in_s = clip_start_in_s - context_pad_in_s
            clip_end_in_s = clip_end_in_s + context_pad_in_s
        clip_dur_in_s = clip_end_in_s - clip_start_in_s
        # Load audio, pad, slice with indexes that account for padding
        load_full_file = True
        if (clip_start_in_s >= 0) and (clip_end_in_s < dfi.total_file_duration_in_s):
            # Attempt to read only the specified excerpt
            myfile = sf.SoundFile(dfi.src_fn)
            if myfile.seekable():
                src_sr = myfile.samplerate
                frame_start = int(np.round(clip_start_in_s * src_sr))
                frames = int(np.round(clip_dur_in_s * src_sr))
                myfile.seek(frame_start)
                y = myfile.read(frames, always_2d=True)
                y = np.mean(y, axis=1)
                load_full_file = False
        if load_full_file:
            # If impossible, read full audio file
            y, src_sr = sf.read(dfi.src_fn, always_2d=True)
            y = np.mean(y, axis=1)
            frame_start = int(np.round(clip_start_in_s * src_sr))
            frames = int(np.round(clip_dur_in_s * src_sr))
            if frame_start < 0:
                y = np.pad(y, [-frame_start, 0])
                frame_start = 0
            if frame_start + frames > len(y):
                y = np.pad(y, [0, frame_start + frames - len(y)])
            y = y[frame_start : frame_start + frames]
        # Resample from src_sr to sr
        y = soxr.resample(y, src_sr, sr).astype(np.float32)
        # If not yet jittered, apply jitter at end via asymmetric zero-padding
        if jitter_via_zero_padding:
            jitter_pad_width = int(np.round(2 * np.abs(jitter_in_s) * sr))
            if jitter_in_s > 0:
                y = np.pad(y, [jitter_pad_width, 0]).astype(np.float32)
            else:
                y = np.pad(y, [0, jitter_pad_width]).astype(np.float32)
        # Zero-pad or trim to length (fixes off by one errors)
        y = self.pad_or_trim_to_len(y, int(dur * sr))
        y = np.nan_to_num(y.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        y = torch.from_numpy(y)
        return y
