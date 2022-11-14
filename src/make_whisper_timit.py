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



    snr = 0 # start with 0 dB 
    print("Starting stimuli generation...") 
    sr = 20_000
    # create trial stim 
    for ix, row in to_run.iterrows():
        print(f"On row {ix} of {len(to_run)}")
        # add wantned data to trial dict
        update_dict(trial_data, row)

        # get signals 
        target_signal = row.signal

        distractor_data = target_timit[target_timit.speaker!=row.speaker].sample(1)
        distractor_signal = distractor_data['signal'].item()

        cue_data = cue_timit[cue_timit.speaker == row.speaker].sample(1)
        cue_signal = cue_data['signal'].item()

        # get whispered signals 
        cue_whisper = make_whispered_speech(cue_signal, sr, eng=eng)
        target_whisper = make_whispered_speech(target_signal, sr, eng=eng)
        distractor_whisper = make_whispered_speech(distractor_signal, sr, eng=eng)
        
        cue, _ = rms_normalize(cue_whisper)
        target_whisper, _ = rms_normalize(target_whisper)
        distractor_whisper, _ = rms_normalize(distractor_whisper)

        # mix target & distractor
        mixture, _ = combine_with_noise(target_whisper, distractor_whisper, snr) # first_scale_factor unused
        mixture, mixture_scale_factor = rms_normalize(mixture)

        # rove cue
        cue = cue * mixture_scale_factor

        # save values for tiral 
        sig_ix = ix % start if start >0 else ix 
        trial_data['signal'][sig_ix] = target_whisper # update signal to only include curr
        trial_data['mixture_signal'].append(mixture)
        trial_data['distractor_signal'].append(distractor_whisper)
        trial_data['cue_signal'].append(cue)
        trial_data['cue_word'].append(cue_data.word.item())
        trial_data['cue_speaker'].append(cue_data.speaker.item())
        trial_data['_original_cue_timit_index'].append(cue_data._original_timit_index.item())
        trial_data['_original_distractor_timit_indices'].append(distractor_data._original_timit_index.item())
        trial_data['distractor_words'].append(distractor_data.word.item())
        trial_data['distractor_speakers'].append(distractor_data.speaker.item())
        trial_data['distractor_conditions'].append('Whispered')
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
    
    out_path = Path('/om2/user/imgriff/datasets/timit/whispered_timit/')
    out_name = out_path / f'timit_whisper_attn_stim_for_model_subset_{args.array_ix:02d}.pdpkl'

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
