import numpy as np 
import soundfile as sf
import librosa
import pandas as pd 
from pathlib import Path
from tqdm.auto import tqdm
import pickle 
import sys 

sys.path.append('../../datasets/spatial_audio_pipeline/spatial_audio_util/')
import util_audio 
from argparse import ArgumentParser
sys.path.append('/om2/user/msaddler/python-packages/msutil')
import util_stimuli

next_pow_2 = lambda x: int(pow(2, np.ceil(np.log2(x))))

def combine_with_noise(clean, noise, snr):
    # get ratio in rms 
    rms_ratio = np.power(10.0, snr / 20.0)
    # remove DC of each signal
    clean = clean - clean.mean()
    noise = noise - noise.mean()
    # get rms of each signal
    clean_rms = np.sqrt(np.mean(np.power(clean, 2)))
    noise_rms = np.sqrt(np.mean(np.power(noise, 2)))
    # scale factor for setting noise to desired SNR 
    scale_factor = clean_rms / (noise_rms * rms_ratio)
    # Blend signals 
    noise = noise * scale_factor
    mixture = clean + noise[:len(clean)]
    return mixture, scale_factors

def rms_normalize(wav, new_rms=0.1, axis=0): 
    wav = wav - wav.mean(axis=axis)
    rms_wav = np.sqrt(np.mean(np.power(wav, 2), axis=axis))
    scale_factor = new_rms / rms_wav
    wav = wav * scale_factor
    return wav, scale_factor

def rms_normalize_db(wav, dBSPL, axis=0): 
    wav = wav - wav.mean(axis=axis)
    rms_wav = np.sqrt(np.mean(np.power(wav, 2), axis=axis))
    new_rms = 20e-6 * np.power(10, dBSPL/20)
    scale_factor = new_rms / rms_wav
    wav = wav * scale_factor
    return wav, scale_factor

            
def main(args):
    np.random.seed(args.array_ix)

    print("Loading source excerpts.") 
    excerpts = pd.read_pickle(args.stim_manifest_path)
    print(args.stim_manifest_path)
    
    with open(args.job_manifest, "rb") as f :
        # read condition dict from pickle
        cond_ix_dict = pickle.load(f)
        condition = cond_ix_dict[args.array_ix][0]
        snr = cond_ix_dict[args.array_ix][1]

    # get distractor column name 
    if 'mandarin' in condition:
        dist_src_col = 'zh_distractor_src_fn'
    elif 'dutch' in condition:
        dist_src_col = 'nl_distractor_src_fn'
    else:
        dist_src_col = 'distractor_fn'
        if 'safe' in args.stim_manifest_path:
            dist_src_col = 'distractor_src_fn'

    cond_id = args.array_ix
    stim_out_path = Path(args.stim_out_path)
    stim_out_dir = stim_out_path / f"condition_{cond_id:02}"
    stim_out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving stimuli to {stim_out_dir.as_posix()}")
    
    SR = 44100
    SPL = 60 # in dB SPL 
    # timing in seconds 
    trial_dur = 4.5
    signal_dur = 2
    isi = 0.5
    win_dur = 10 # ms 
    new_rms = 0.02 # is 60dB in  amplitude e.g 20e-6pa * 10**(60/20)
    mixture_onset = int((isi + signal_dur) * SR) # in frames

    if 'safe' in args.stim_manifest_path:
        cue_fn_str = 'cue_src_fn'
        target_fn_str = 'src_fn'
    else:
        cue_fn_str = "cue_fn"
        target_fn_str = "target_fn"

    print("Starting stimuli generation...") 

    # create trial stim using existing excerpt file names 
    fname_array = excerpts[[cue_fn_str, target_fn_str, dist_src_col]].values

    for ix, (cue_fn, target_fn, distractor_fn) in enumerate(tqdm(fname_array)):
        # init output signal 
        trial_signal = np.zeros((int(SR * trial_dur)),dtype=np.float32)
        # load already cut/resampled/windowed signals 
        cue_signal, cue_sr = librosa.load(cue_fn, sr=SR)
        target_signal, target_sr = librosa.load(target_fn, sr=SR)
        distractor_signal, dist_sr = librosa.load(distractor_fn, sr=SR)

        # normalize signals to same level for safe mixing (is not 100% necessary) 
        cue_signal = util_audio.set_dBSPL(cue_signal, SPL)
        target_signal = util_audio.set_dBSPL(target_signal, SPL)
        distractor_signal = util_audio.set_dBSPL(distractor_signal, SPL)
        
        if snr == 'inf':
            mixture = target_signal 
        else:
            mixture = util_audio.combine_signal_and_noise(target_signal, distractor_signal, snr)
        mixture = util_audio.ramp_hann(mixture, win_dur, samplerate=SR)
        mixture, mixture_scale_factor = rms_normalize_db(mixture, SPL)
        # set cue to same level as target post scaling
        cue_signal = util_audio.ramp_hann(cue_signal, win_dur, samplerate=SR)
        cue_signal = util_audio.set_dBSPL(cue_signal, SPL)
        cue_signal *= mixture_scale_factor

        # add parts to trial signal
        trial_signal[ : len(cue_signal)] += cue_signal  
        trial_signal[mixture_onset : ] += mixture
        trial_signal = util_audio.set_dBSPL(trial_signal, SPL)

        # save trial signal
        # f string with digint padded to 3 places
        out_name = stim_out_dir / f'{ix:03}.wav'
        sf.write(out_name, trial_signal, samplerate=SR)

    print("Finished stimuli generation.") 

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
    "--array_ix",
    default=0,
    type=int,
    help="SLURM job array ix.",
    )
    parser.add_argument(
    "--stim_manifest_path",
    default='/om/user/imgriff/datasets/human_distractor_language_2024/human_expmt_manifest_w_transcripts.pdpkl',
    type=str,
    help="Path to manifest of trial source stimuli.",
    )
    parser.add_argument(
    "--job_manifest",
    default='/om/user/imgriff/datasets/human_distractor_language_2024/human_distractor_language_cond_map.pkl',
    type=str,
    help="Path to manifest holding condition to job ix mapping.",
    )
    parser.add_argument(
    "--stim_out_path",
    default='/om/user/imgriff/datasets/human_distractor_language_2024/sounds',
    type=str,
    help="Path to dir to save stimuli.",
    )
    
    args = parser.parse_args()

    main(args)