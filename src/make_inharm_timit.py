import pandas as pd
import pickle
import numpy as np 
from pathlib import Path
from tqdm import tqdm
from argparse import ArgumentParser

import matlab.engine
import sys 
sys.path.append('/om2/user/msaddler/python-packages/msutil')
import util_stimuli

np.random.seed(0)


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
    print("Starting MATLAB engine...") 
    eng = matlab.engine.start_matlab();
    eng.addpath('/om2/user/imgriff/projects/msSTRAIGHT/Sinusoidal_Straight_Toolbox_v1.0/');
    eng.addpath('/om2/user/imgriff/projects/msSTRAIGHT/');

    print("Loading TIMIT excerpts.") 
    timit_path = '/om2/user/imgriff/datasets/timit/safe_sentences_timit_wsn_compatible_0.1rms.pdpkl'

    timit_excerpts = pd.read_pickle(timit_path)
    timit_excerpts = timit_excerpts.rename(columns={"_full_dataset_index":"_original_timit_index"})
    # parse target & rest 

    target_timit = timit_excerpts[timit_excerpts.word_int != 0]
    target_timit.reset_index(inplace=True, drop=True)
    
    # get subset of array to run
    start = args.array_ix * 50 
    stop = start + 50 
    if abs(len(target_timit) - stop) < 10:
        stop = len(target_timit) 
    to_run = target_timit.iloc[start:stop]
    # target_sentences = target_timit.sentence_id.unique()

    # filter cues 
    cue_timit = timit_excerpts[timit_excerpts.word_int == 0]
    # cue_timit = timit_excerpts[(timit_excerpts.word_int == 0) & (~timit_excerpts.sentence_id.str.contains('1|2'))]
    
    # Generate trial by trial stim    
    trial_data = {'signal':[],
                  'speaker': [],
                  'speaker_sex': [],
                  'sentence_type': [],
                  'word_int': [],
                  'mixture_signal':[],
                  'cue_signal': [],
                  'cue_speaker': [],
                  'cue_word': [],
                  'distractor_signal':[],
                  '_original_distractor_timit_indices':[],
                  '_original_cue_timit_index':[],
                  'distractor_words':[],
                  'distractor_speakers':[],
                  'distractor_conditions':[],
                  'distractor_sex':[],
                  'snrs':[]}


    # use params from Popham et al. 2018
    harm_nums = np.arange(1, 31)
    jitter_amount = 0.3
    min_diff = 30.0

    snr = 0 # start with 0 dB 
    print("Starting stimuli generation...") 

    # create trial stim 
    for ix, row in tqdm(to_run.iterrows()):
        # add wantned data to trial dict
        update_dict(trial_data, row)

        # get signals 
        target_signal = row.signal

        distractor_data = target_timit[target_timit.speaker!=row.speaker].sample(1)
        distractor_signal = distractor_data['signal'].item()

        cue_data = cue_timit[cue_timit.speaker == row.speaker].sample(1)
        cue_signal = cue_data['signal'].item()

        ## get min f0s 
        min_f0s = []
        for source in [cue_signal, target_signal, distractor_signal]:
            f0_contour = get_f0_contour(source, 20_000, eng=eng)
            min_contour_f0 = f0_contour[f0_contour!=0].min()
            min_f0s.append(min_contour_f0)

        min_f0 = min(min_f0s)

        # Make harmonic jitter 
        jitter_pattern = eng.make_jittered_speech_harmonics(
            matlab.double([min_f0]),
            matlab.double(harm_nums.tolist()),
            matlab.double([jitter_amount]),
            matlab.double([min_diff]))

        jitter_pattern = np.array(jitter_pattern).reshape([-1]).astype(float)

        # get inharmonic signals 
        cue_inharm = impose_inharmonic_jitter_pattern(cue_signal, 20_000, jitter_pattern, eng=eng)
        target_inharm = impose_inharmonic_jitter_pattern(target_signal, 20_000, jitter_pattern, eng=eng)
        distractor_inharm = impose_inharmonic_jitter_pattern(distractor_signal, 20_000, jitter_pattern, eng=eng) 

        cue, _ = rms_normalize(cue_inharm)
        target_inharm, _ = rms_normalize(target_inharm)
        distractor_inharm, _ = rms_normalize(distractor_inharm)
        
        # mix target & distractor
        mixture, _ = combine_with_noise(target_inharm, distractor_inharm, snr) # first_scale_factor unused
        mixture, mixture_scale_factor = rms_normalize(mixture)

        # rove cue
        cue = cue * mixture_scale_factor

        # save values for tiral 
        sig_ix = ix % start if start > 0 else ix 
        trial_data['signal'][sig_ix] = target_inharm
        trial_data['mixture_signal'].append(mixture)
        trial_data['distractor_signal'].append(distractor_inharm)
        trial_data['cue_signal'].append(cue)
        trial_data['cue_word'].append(cue_data.word.item())
        trial_data['cue_speaker'].append(cue_data.speaker.item())
        trial_data['_original_cue_timit_index'].append(cue_data._original_timit_index.item())
        trial_data['_original_distractor_timit_indices'].append(distractor_data._original_timit_index.item())
        trial_data['distractor_words'].append(distractor_data.word.item())
        trial_data['distractor_speakers'].append(distractor_data.speaker.item())
        trial_data['distractor_conditions'].append('inharmonic')
        trial_data['distractor_sex'].append(distractor_data.speaker_sex.item())
        trial_data['snrs'].append(snr)

    # convert to pandas 
    trial_data = pd.DataFrame(trial_data)
    print("Finished stimuli generation.") 

    # get word to class ix mapping 
    word_and_speaker_encodings = pickle.load( open( "/om2/user/imgriff/projects/End-to-end-ASR-Pytorch/word_and_speaker_encodings_jsinv3.pckl", "rb" )) 
    class_map = word_and_speaker_encodings['word_idx_to_word']
    word_to_ix_map = {val:key for key,val in class_map.items()}
    
    def get_ix_from_words(words):
        if not isinstance(words, list):
            return words
        else:
            return [word_to_ix_map[word] for word in words]

    trial_data['distractor_word_ints'] = trial_data['distractor_words'].apply(get_ix_from_words)
    
    out_path = Path('/om2/user/imgriff/datasets/timit/inharmonic_timit/')
    out_name = out_path / f'timit_inharm_attn_stim_for_model_subset_{args.array_ix:02d}.pdpkl'

    print(f"Saving stimuli to: {out_name.as_posix()}") 
          
    trial_data.to_pickle(out_name)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
    "--array_ix",
    default=0,
    type=int,
    help="SLURM job array ix ",
    )
    
    args = parser.parse_args()

    main(args)
