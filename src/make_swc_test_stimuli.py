from pathlib import Path
import pandas as pd 
import librosa


from IPython.display import Audio
from scipy.io.wavfile import read, write
import numpy as np 
import soundfile as sf
import soxr
from tqdm.auto import tqdm

pd_path = Path('/om2/user/imgriff/datasets/spatial_audio_pipeline/assets/human_attn_experiment_v00/')
trial_df = pd.read_pickle(pd_path / 'full_eval_trial_manifest.pdpkl')

trial_df['cue_clip_dur_in_s'] = trial_df['cue_end_in_s'] - trial_df['cue_start_in_s']

# rename cue_start_in_s to clip_start_in_s
trial_df.rename(columns={'cue_start_in_s':'cue_clip_start_in_s'}, inplace=True)
# same for end 
trial_df.rename(columns={'cue_end_in_s':'cue_clip_end_in_s'}, inplace=True)
trial_df.rename(columns={'cue_fn':'cue_src_fn'}, inplace=True)

target_path = Path('/om/user/imgriff/datasets/spatial_audio_pipeline/assets/human_attn_experiment_v00/target_excerpts')
target_path.mkdir(exist_ok=True, parents=True)

cue_path = Path('/om/user/imgriff/datasets/spatial_audio_pipeline/assets/human_attn_experiment_v00/cue_excerpts')
cue_path.mkdir(exist_ok=True, parents=True)

distractor_path = Path('/om/user/imgriff/datasets/spatial_audio_pipeline/assets/human_attn_experiment_v00/distractor_excerpts')
distractor_path.mkdir(exist_ok=True, parents=True)

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
    
def get_excerpt(dfi, dur=3.0, sr=50000, pad_with_context=True, jitter_fraction=0):
    """
    This function loads an audio file and excerpts a clip with the specified
    duration. Target durations that exceed clip boundaries are handled with
    zero-padding (applied to all signals but sliced away when not needed).
    This function also handles resampling (via soxr) and re-scaling.
    """
    jitter_in_s = 0
    jitter_via_zero_padding = True
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
    load_full_file = True
    total_file_duration_in_s = librosa.get_duration(filename=dfi.src_fn)
    if (clip_start_in_s >= 0) and (clip_end_in_s < total_file_duration_in_s):
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
    y = pad_or_trim_to_len(y, int(dur * sr))
    y = np.nan_to_num(y.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    return y


distractor_df = trial_df.filter(regex='distractor_')
distractor_df.columns = distractor_df.columns.str.replace('distractor_', '')

# do the same for cue_df
cue_df = trial_df.filter(regex='cue_')
cue_df.columns = cue_df.columns.str.replace('cue_', '')

sr = 44100
duration = 2.0

for ix in tqdm(range(len(trial_df)), total=len(trial_df)):
    # get row from trial_df
    dfi = trial_df.iloc[ix]
    # get excerpt of target
    target_excerpt = get_excerpt(dfi, dur=duration, jitter_fraction=0)
    # get excerpt of cue
    cue_excerpt = get_excerpt(cue_df.iloc[ix], dur=duration, jitter_fraction=0)
    # get excerpt of distractor
    distractor_excerpt = get_excerpt(distractor_df.iloc[ix], dur=duration, jitter_fraction=0)
    # write target excerpt to file
    target_fn = target_path / f"{dfi.word}_{dfi.client_id}.wav"
    write(target_fn, sr, target_excerpt)
    # write cue excerpt to file
    cue_fn = cue_path / f"{dfi.cue_word}_{dfi.cue_client_id}.wav"
    write(cue_fn, sr, cue_excerpt)
    # write distractor excerpt to file
    distractor_fn = distractor_path / f"{dfi.distractor_word}_{dfi.distractor_client_id}.wav"
    write(distractor_fn, sr, distractor_excerpt)
    # update manifest with new file names
    trial_df.loc[ix, 'src_fn'] = str(target_fn)
    trial_df.loc[ix, 'cue_fn'] = str(cue_fn)
    trial_df.loc[ix, 'distractor_fn'] = str(distractor_fn)

trial_df.to_pickle(pd_path / 'full_eval_trial_manifest_new_fnames.pdpkl')


