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
import matlab.engine
sys.path.append('/om2/user/msaddler/python-packages/msutil')
import util_stimuli


def impose_inharmonic_jitter_pattern(y, sr, jitter, eng=None):
    x = matlab.double(y.reshape([-1, 1]).tolist())
    fs = matlab.double([sr])
    jitter_pattern = matlab.double(np.array(jitter).reshape([-1, 1]).tolist())
    
    y_resynth_matlab = eng.STRAIGHT_impose_inharmonic_jitter_pattern(
        x,
        fs,
        jitter_pattern)
    y_resynth = np.array(y_resynth_matlab).astype(np.float32).reshape([-1])
    NAN_IDX = np.isnan(y_resynth)
    if NAN_IDX.sum() > 0:
        print("Replacing {} NaN values (of {}) with zeros".format(NAN_IDX.sum(), NAN_IDX.shape[0]))
        y_resynth[NAN_IDX] = 0
    npad = int((y.shape[0] - y_resynth.shape[0]) / 2)
    y_resynth = np.pad(y_resynth, [npad, npad])
    return y_resynth


def make_whispered_speech(y, sr, fail_threshold=0.25, eng=None):
    y_resynth = eng.STRAIGHT_whispered_speech(
        matlab.double(y.reshape([-1, 1]).tolist()),
        matlab.double([sr]),
        nargout=1)
    y_resynth = np.array(y_resynth).astype(np.float32).reshape([-1])
    NAN_IDX = np.isnan(y_resynth)
    if NAN_IDX.sum() > 0:
        print("Replacing {} NaN values (of {}) with zeros".format(NAN_IDX.sum(), NAN_IDX.shape[0]))
        y_resynth[NAN_IDX] = 0
    y_resynth = util_stimuli.pad_or_trim_to_len(y_resynth, y.shape[0], mode='end')
    if (util_stimuli.rms(y_resynth) == 0) or (NAN_IDX.sum() / NAN_IDX.shape[0] > fail_threshold):
        print("STRAIGHT FAILURE --> returning input instead")
        y_resynth = y
    return y_resynth


def get_f0_contour(y, sr, eng=None):
    x = matlab.double(y.reshape([-1, 1]).tolist())
    fs = matlab.double([sr])
    f0_contour_matlab = eng.STRAIGHT_analysis_get_f0(x, fs)
    f0_contour = np.array(f0_contour_matlab).astype(np.float32).reshape([-1])
    NAN_IDX = np.isnan(f0_contour)
    if NAN_IDX.sum() > 0:
        print("Replacing {} NaN values (of {}) with zeros".format(NAN_IDX.sum(), NAN_IDX.shape[0]))
        f0_contour[NAN_IDX] = 0
    return f0_contour


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
    return mixture, scale_factor


def rms_normalize(wav, new_rms=0.1, axis=0): 
    wav = wav - wav.mean(axis=axis)
    rms_wav = np.sqrt(np.mean(np.power(wav, 2), axis=axis))
    scale_factor = new_rms / rms_wav
    wav = wav * scale_factor
    return wav, scale_factor


def update_dict(dict_to_update, dict_with_vals):
    for key,value in dict_with_vals.items():
        if key not in dict_to_update:
            dict_to_update[key] = [value]
        else:
            dict_to_update[key].append(value)
            
def main(args):
    np.random.seed(args.array_ix)

    print("Starting MATLAB engine...") 
    eng = matlab.engine.start_matlab();
    eng.addpath('/om2/user/imgriff/projects/msSTRAIGHT/Sinusoidal_Straight_Toolbox_v1.0/');
    eng.addpath('/om2/user/imgriff/projects/msSTRAIGHT/');

    print("Loading source excerpts.") 
    excerpts = pd.read_pickle(args.stim_manifest_path)
    
    with open(args.job_manifest, "rb") as f :
        # read condition dict from pickle
        cond_ix_dict = pickle.load(f)
        target_harm = cond_ix_dict[args.array_ix]['target_harmonicity']
        dist_harm = cond_ix_dict[args.array_ix]['distractor_harmonicity']
        cond_id = cond_ix_dict[args.array_ix]['condition']

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

    # use params from Popham et al. 2018
    dummy_jitter_pattern = np.zeros(30).astype(float) # for harmonic condition
    harm_nums = np.arange(1, 31) # 1-31 for 30 harmonics compat with matlab index
    jitter_amount = 0.3 
    min_diff = 30.0

    snr = 0 # start with 0 dB 
    print("Starting stimuli generation...") 

    # get subset of array to run
    n_conds = 12 
    start = (args.array_ix % n_conds) * args.n_trials_per_job
    stop = start + args.n_trials_per_job 
    if abs(len(excerpts) - stop) < 10:
        stop = len(excerpts) 
    excerpts = excerpts.iloc[start:stop]

    print(f"Running condition {cond_id}: {target_harm} target {dist_harm} distractor for trials {start} thru {stop}")

    # create trial stim using existing excerpt file names 
    fname_array = excerpts[["cue_fn", "target_fn", "distractor_fn"]].values

    for ix, (cue_fn, target_fn, distractor_fn) in enumerate(tqdm(fname_array)):
        # account for job ix offset 
        ix = ix + start 
        # init output signal 
        trial_signal = np.zeros((int(SR * trial_dur)),dtype=np.float32)
        # load already cut/resampled/windowed signals 
        cue_signal, cue_sr = librosa.load(cue_fn, sr=SR)
        target_signal, target_sr = librosa.load(target_fn, sr=SR)
        distractor_signal, dist_sr = librosa.load(distractor_fn, sr=SR)

        ## get min f0s 
        min_f0s = []
        for source in [cue_signal, target_signal, distractor_signal]:
            f0_contour = get_f0_contour(source, SR, eng=eng)
            min_contour_f0 = f0_contour[f0_contour > 0 ].min()
            min_f0s.append(min_contour_f0)
        print(f"Mins on trial {ix} --> {min_f0s}")

        # Clip bad F0 estimates ie. f0 below reasonable vocal range 
        min_f0 = np.array(min_f0s).clip(60).min()
    
        if target_harm == 'inharmonic' or dist_harm == 'inharmonic':
            # Get inharmonic jitter pattern
            jitter_pattern = eng.make_jittered_speech_harmonics(
                matlab.double([min_f0]),
                matlab.double(harm_nums.tolist()),
                matlab.double([jitter_amount]),
                matlab.double([min_diff]))
            jitter_pattern = np.array(jitter_pattern).reshape([-1]).astype(float)
        
        # determine target manipulation (for cue and target)
        if target_harm == 'inharmonic':
            target_jitter = jitter_pattern
        elif target_harm == 'harmonic':
            target_jitter = dummy_jitter_pattern

        # apply target manipulation (for cue and target)
        if target_harm == 'whispered':
            cue_signal = make_whispered_speech(cue_signal, SR, eng=eng)
            target_signal = make_whispered_speech(target_signal, SR, eng=eng)
        else:
            cue_signal = impose_inharmonic_jitter_pattern(cue_signal, SR, target_jitter, eng=eng)
            target_signal = impose_inharmonic_jitter_pattern(target_signal, SR, target_jitter, eng=eng)
       
        # normalize signals to same level for safe mixing (is not 100% necessary) 
        cue_signal= util_audio.set_dBSPL(cue_signal, SPL)
        target_signal= util_audio.set_dBSPL(target_signal, SPL)


        ## determine distractor manipulation
        # if not None
        if dist_harm: 
            if dist_harm == 'inharmonic':
                dist_jitter = jitter_pattern
            elif dist_harm == 'harmonic':
                dist_jitter = dummy_jitter_pattern
            # apply distractor manipulation
            if dist_harm == 'whispered':
                distractor_signal = make_whispered_speech(distractor_signal, SR, eng=eng)
            else:
                distractor_signal = impose_inharmonic_jitter_pattern(distractor_signal, SR, dist_jitter, eng=eng)
            distractor_signal= util_audio.set_dBSPL(distractor_signal, SPL)

            # mix target & distractor
            mixture, _ = combine_with_noise(target_signal, distractor_signal, snr) # first_scale_factor unused
            mixture, mixture_scale_factor = rms_normalize(mixture, new_rms=new_rms)
            # rove cue to match new target level
            cue_signal *= mixture_scale_factor

        else:
            mixture = target_signal
        
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
    "--n_trials_per_job",
    default=30,
    type=int,
    help="Number of trials to generate per job.",
    )
    parser.add_argument(
    "--stim_manifest_path",
    default='/om/user/imgriff/datasets/human_swc_popham_exmpt_2024/source_stim_meta_manifest.pdpkl',
    type=str,
    help="Path to manifest of trial source stimuli.",
    )
    parser.add_argument(
    "--job_manifest",
    default='swc_popham_exmpt_2024_stim_gen_conds.pkl',
    type=str,
    help="Path to manifest holding condition to job ix mapping.",
    )
    parser.add_argument(
    "--stim_out_path",
    default='/om/user/imgriff/datasets/human_swc_popham_exmpt_2024/sounds',
    type=str,
    help="Path to dir to save stimuli.",
    )
    
    args = parser.parse_args()

    main(args)