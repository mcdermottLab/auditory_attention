import numpy as np 
import soundfile as sf
import librosa
import pandas as pd 
from pathlib import Path
from tqdm.auto import tqdm
import pickle 
import sys 
import h5py
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

def update_dict(dict_to_update, dict_with_vals):
    for key,value in dict_with_vals.items():
        if key not in dict_to_update:
            dict_to_update[key] = [value]
        else:
            dict_to_update[key].append(value)

def process_signal(signal, sig_len_frames, window_dur, SR, SPL):
    # ramp edges
    signal = util_audio.pad_or_trim_to_len(signal, sig_len_frames, mode='both')
    signal = util_audio.ramp_hann(signal, window_dur, samplerate=SR)
    signal = util_audio.set_dBSPL(signal, SPL)
    return signal
            
def main(args):
    np.random.seed(args.array_ix)

    print("Starting MATLAB engine...") 
    eng = matlab.engine.start_matlab();
    eng.addpath('/om2/user/imgriff/projects/msSTRAIGHT/Sinusoidal_Straight_Toolbox_v1.0/');
    eng.addpath('/om2/user/imgriff/projects/msSTRAIGHT/');

    print("Loading source excerpts.") 
    manifest = pd.read_pickle('/om/user/imgriff/datasets/human_word_rec_SWC_2024/human_cue_target_distractor_df_w_meta_transcripts.pdpkl')
    word_dict = pickle.load(open("/om2/user/imgriff/datasets/commonvoice_9/en/cv_800_word_label_to_int_dict.pkl", 'rb'))
    manifest['word_int_label'] = manifest['word'].replace(word_dict)
    ## add original manifest index as a column
    manifest['orig_df_ix'] = manifest.index
    start_idx = args.array_ix * args.num_manifest_rows
    end_idx = min(start_idx + args.num_manifest_rows, len(manifest))
    wanted_manifest = manifest.iloc[start_idx:end_idx]


    stim_out_dir = Path(args.stim_out_path) / 'model_eval_h5s'
    stim_out_dir.mkdir(parents=True, exist_ok=True)
    output_file_name = stim_out_dir / f'model_popham_stim_separated_manifest_rows_{start_idx}-{end_idx}.h5'
    print(f"Saving stimuli to {output_file_name.as_posix()}")
    
    SR = 44100
    SPL = 60 # in dB SPL 
    # timing in seconds 
    signal_dur = 2.5 # add context to crop to 2s post cochleagram 
    win_dur = 10 # ms 
    sig_len_frames = int(signal_dur * SR)

    # use params from Popham et al. 2018
    dummy_jitter_pattern = np.zeros(30).astype(float) # for harmonic condition
    harm_nums = np.arange(1, 31) # 1-31 for 30 harmonics compat with matlab index
    jitter_amount = 0.3 
    min_diff = 30.0

    ## init kest for h5 file 
    signal_keys  = [
        'harmonic_cue_signal',
        'harmonic_target_signal',
        'harmonic_distractor_signal',
        'inharmonic_cue_signal',
        'inharmonic_target_signal',
        'inharmonic_distractor_signal',
        'whispered_cue_signal',
        'whispered_target_signal',
        'whispered_distractor_signal',
                ]
    label_keys = ['word_int_label', 'sr', 'orig_df_ix', 'distractor_sex_int']
    signal_len = int(SR * signal_dur)
    n_examples = len(wanted_manifest) * 2 # for same and different sex distractors
    example_ix = 0 

    # setup timer for time per trial 
    print("Setting up timer for time per trial.")
    import time
    start_time = time.time()


    # sort manifest by word 
    if not output_file_name.exists():
        with h5py.File(output_file_name, 'w-') as f:
            for key in signal_keys:
                f.create_dataset(key, shape=[n_examples, signal_len], dtype=np.float32)
            for key in label_keys:
                f.create_dataset(key, shape=[n_examples], dtype=np.int32)

    print("Starting stimuli generation...") 
    with h5py.File(output_file_name, 'r+') as f:
        for row in tqdm(wanted_manifest.itertuples(), total=len(wanted_manifest)):
            # load cue and talker signal once 
            cue_signal, _ = librosa.load(row.excerpt_cue_src_fn, sr=SR)
            target_signal, _ = librosa.load(row.excerpt_src_fn, sr=SR)
            for sex_cond in ['same', 'diff']:
                if f['sr'][example_ix] != 0:
                    print(f"Skipping {example_ix} as it already exists.")
                    example_ix += 1
                    continue
                # select distractor signal 
                if sex_cond == 'same':
                    distractor_signal, _ = librosa.load(row.excerpt_same_sex_distractor_1_src_fn, sr=SR)
                    dist_sex_int = 0
                elif sex_cond == 'diff':
                    distractor_signal, _ = librosa.load(row.excerpt_diff_sex_distractor_1_src_fn, sr=SR)
                    dist_sex_int = 1
                ## get min f0s for pairing to determine jitter pattern
                min_f0s = []
                for source in [cue_signal, target_signal, distractor_signal]:
                    f0_contour = get_f0_contour(source, SR, eng=eng)
                    min_contour_f0 = f0_contour[f0_contour > 0 ].min()
                    min_f0s.append(min_contour_f0)
                print(f"f0s on trial {example_ix} --> {min_f0s}")

                # Clip bad F0 estimates ie. f0 below reasonable vocal range 
                min_f0 = np.array(min_f0s).clip(60).min()
                
                for harmonicity in ['harmonic', 'inharmonic', 'whispered']:
                    if harmonicity == 'inharmonic':
                        # Get inharmonic jitter pattern
                        jitter_pattern = eng.make_jittered_speech_harmonics(
                            matlab.double([min_f0]),
                            matlab.double(harm_nums.tolist()),
                            matlab.double([jitter_amount]),
                            matlab.double([min_diff]))
                        jitter_pattern = np.array(jitter_pattern).reshape([-1]).astype(float)
                    elif harmonicity == 'harmonic':
                        jitter_pattern = dummy_jitter_pattern
         
                    # apply target manipulation (for cue and target)
                    if harmonicity == 'whispered':
                        manipulated_cue = make_whispered_speech(cue_signal, SR, eng=eng)
                        manipulated_target = make_whispered_speech(target_signal, SR, eng=eng)
                        manipulated_distractor = make_whispered_speech(distractor_signal, SR, eng=eng)
                    else:
                        manipulated_cue = impose_inharmonic_jitter_pattern(cue_signal, SR, jitter_pattern, eng=eng)
                        manipulated_target = impose_inharmonic_jitter_pattern(target_signal, SR, jitter_pattern, eng=eng)
                        manipulated_distractor = impose_inharmonic_jitter_pattern(distractor_signal, SR, jitter_pattern, eng=eng)

                    # normalize rms
                    manipulated_cue = process_signal(manipulated_cue, sig_len_frames, win_dur, SR, SPL)
                    manipulated_target = process_signal(manipulated_target, sig_len_frames, win_dur, SR, SPL)
                    manipulated_distractor = process_signal(manipulated_distractor, sig_len_frames, win_dur, SR, SPL)

                    # save to h5 file
                    f[f'{harmonicity}_cue_signal'][example_ix] = manipulated_cue
                    f[f'{harmonicity}_target_signal'][example_ix] = manipulated_target
                    f[f'{harmonicity}_distractor_signal'][example_ix] = manipulated_distractor

                # save label
                f['word_int_label'][example_ix] = row.word_int_label
                f['sr'][example_ix] = SR
                f['orig_df_ix'][example_ix] = row.orig_df_ix
                f['distractor_sex_int'][example_ix] = dist_sex_int

                # print time per tiral 
                if example_ix % 100 == 0:
                    print(f"Finished {example_ix} examples.")
                    elapsed_time = time.time() - start_time
                    print(f"Elapsed time per trial: {elapsed_time / (example_ix + 1)}")

                # increment example index after each sex condition per example
                example_ix += 1

    print("Finished stimuli generation.") 

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
    "--array_ix",
    default=0,
    type=int,
    help="Array index for parallelization.",
    )
    parser.add_argument(
    "--num_manifest_rows",
    default=30,
    type=int,
    help="Number of rows from parent manifest to process per array job.",
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
    default='/om/user/imgriff/datasets/human_swc_popham_exmpt_2024/',
    type=str,
    help="Path to dir to save stimuli.",
    )
    
    args = parser.parse_args()

    main(args)