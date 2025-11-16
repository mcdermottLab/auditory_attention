import argparse
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
import soxr

def main(args):
    expmt_dir = Path('/om/user/imgriff/datasets/human_word_rec_SWC_2024/')
    manifest = pd.read_pickle('/om/user/imgriff/datasets/human_word_rec_SWC_2024/human_cue_target_distractor_df_w_meta_transcripts.pdpkl')

    with open(expmt_dir/"human_attn_expmt_cond_map.pkl", "rb") as f :
        # read condition dict from pickle
        cond_ix_dict = pickle.load(f)
    
    with open(expmt_dir/"human_attn_expmt_word_key.pkl", "rb") as f :
        word_key_dict = pickle.load(f)
    word_to_ix = {word: ix for ix, word in word_key_dict.items()}

    SR = 44100
    dBSPL = 60
    # timing in seconds 
    trial_dur = 4.5
    signal_dur = 2
    isi = 0.5
    win_dur = 10 # ms 
    mixture_onset = int((isi + signal_dur) * SR) # in frames
    sig_len_frames = int(signal_dur * SR)
    # get bg conds
    cond_ix = args.array_id
    condition, snr = cond_ix_dict[cond_ix]
    print(f"Generating stimuli for condition {cond_ix}: {condition} at SNR {snr} dB")
    ## will write all possilbe stim combinations to disk - sample in prolific experiment logic 
    # sort manifest by word - orders output examples in 0-(Vocab - 1) in a-z order
    wanted_manifest = manifest.sort_values('word')

    target_f0s = np.zeros(len(wanted_manifest))
    distractor_f0s = np.zeros(len(wanted_manifest))
    for ix, row in enumerate(tqdm(wanted_manifest.itertuples(), total=len(wanted_manifest))):
        trial_signal = np.zeros((int(SR * trial_dur)),dtype=np.float32)
        # are already cut/resampled/windowed
        cue_wav, _ = librosa.load(row.excerpt_cue_src_fn, sr=SR)
        target_wav, _ = librosa.load(row.excerpt_src_fn, sr=SR)
        # get target f0 
        target_f0 = util_audio.get_avg_f0(target_wav, SR, fmin=80, fmax=300)
        target_f0s[ix] = target_f0
        # get distractor signal
        if condition == 'SILENCE':
            bg_signal = np.zeros_like(target_wav)

        elif "1-talker" in condition:
            if "1-talker-english" in condition:
                if 'same' in condition:
                    bg_signal, _ = librosa.load(row.excerpt_same_sex_distractor_1_src_fn, sr=SR)
                elif 'diff' in condition:
                    bg_signal,_ = librosa.load(row.excerpt_diff_sex_distractor_1_src_fn, sr=SR)
            elif "1-talker-mandarin" in condition:
                if 'same' in condition:
                    bg_signal, _ = librosa.load(row.same_sex_zh_distractor_full_path, sr=SR)
                elif 'diff' in condition:
                    bg_signal, _ = librosa.load(row.diff_sex_zh_distractor_full_path, sr=SR)
            elif "1-talker-dutch" in condition:
                if 'same' in condition:
                    bg_signal, _ = librosa.load(row.same_sex_nl_distractor_full_path, sr=SR)
                elif 'diff' in condition:
                    bg_signal, _ = librosa.load(row.diff_sex_nl_distractor_full_path, sr=SR)
            bg_f0 = util_audio.get_avg_f0(bg_signal, SR, fmin=80, fmax=300)
            distractor_f0s[ix] = bg_f0
            # pad or cut to length
            bg_signal = util_audio.pad_or_trim_to_len(bg_signal, sig_len_frames, mode='both')
        
        elif '2-talker' in condition:
            # take one same and one diff
            same_sex_sig, _ = librosa.load(row.excerpt_same_sex_distractor_1_src_fn, sr=SR)
            diff_sex_sig, _ = librosa.load(row.excerpt_diff_sex_distractor_1_src_fn, sr=SR)
            bg_signal = util_audio.combine_signal_and_noise(same_sex_sig, diff_sex_sig, 0)
       
        elif '4-talker' in condition:
            ## Use list comps for this condition
            dist_signals = [librosa.load(getattr(row,col), sr=SR)[0] 
                                for col in ['excerpt_same_sex_distractor_1_src_fn',
                                             'excerpt_same_sex_distractor_2_src_fn',
                                             'excerpt_diff_sex_distractor_1_src_fn',
                                             'excerpt_diff_sex_distractor_2_src_fn']
                            ]
            # set each distractor to same level 
            dist_signals = [util_audio.set_dBSPL(sig, dBSPL) for sig in dist_signals]
            bg_signal = np.sum(dist_signals, axis=0)

        # handle noise distractors 
        elif "issn" in condition:
            bg_signal = util_audio.spectrally_matched_noise(target_wav, SR)
        elif "mus" in condition:
            bg_signal, _ = librosa.load(row.music_full_path, sr=SR)        
        elif "babble" in condition:
            bg_signal, _ = librosa.load(row.babble_full_path, sr=SR)
        elif 'aaspcasa' in condition:
            bg_signal, _ = librosa.load(row.natural_scene_full_path, sr=SR)

        # make sure bg signal is right length 
        bg_signal = util_audio.pad_or_trim_to_len(bg_signal, sig_len_frames)
        # ramp onset and offset of bg signal to remove clicks/pops
        bg_signal = util_audio.ramp_hann(bg_signal, win_dur, samplerate=SR)

        # combine target and bg signal
        if bg_signal.sum() != 0:
            mixture = util_audio.combine_signal_and_noise(target_wav, bg_signal, snr)
        else:
            mixture = target_wav
        # ramp and set level of mixture
        mixture = util_audio.ramp_hann(mixture, win_dur, samplerate=SR)
        mixture = util_audio.set_dBSPL(mixture, dBSPL)
        # set level of cue 
        cue_wav = util_audio.set_dBSPL(cue_wav, dBSPL)

        # add cue to trial signal
        trial_signal[:len(cue_wav)] += cue_wav    
        trial_signal[mixture_onset:] += mixture
        trial_signal = util_audio.set_dBSPL(trial_signal, dBSPL)

        ### save trial signal ###
        # get target sex for fname 
        targ_sex_initial = row.gender[0]
        word_ix = word_to_ix[row.word]
        # f string with digit padded to 3 places
        out_name = expmt_dir / "sounds" / f"condition_{cond_ix:02}"/ f'{targ_sex_initial}_{word_ix:03}.wav'
        # make outname directory if it doesn't exist
        out_name.parent.mkdir(parents=True, exist_ok=True)
        sf.write(out_name, trial_signal, samplerate=SR)
    
    # save f0_manifest just once (should be same across SNRs)
    if '1-talker' in condition and snr == 0:
        df_ixs = wanted_manifest.index
        lang_string = [lang for lang in ['english', 'mandarin', 'dutch'] if lang in condition][0]
        f0_manifest = pd.DataFrame({'df_ixs': df_ixs,
                                    'target_f0': target_f0s,
                                    f'{lang_string}_distractor_f0': distractor_f0s})
        f0_manifest.to_pickle(expmt_dir / f"{lang_string}_1_distractor_f0_manifest.pdpkl")
            
            
            

if __name__ == "__main__":
    # get array id from argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--array_id', type=int, required=True)
    args = parser.parse_args()
    main(args)

