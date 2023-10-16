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

def rms_normalize_db(wav, dBSPL, axis=0): 
    wav = wav - wav.mean(axis=axis)
    rms_wav = np.sqrt(np.mean(np.power(wav, 2), axis=axis))
    new_rms = 20e-6 * np.power(10, dBSPL/20)
    scale_factor = new_rms / rms_wav
    wav = wav * scale_factor
    return wav, scale_factor

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
    y = util_audio.pad_or_trim_to_len(y, int(dur * sr))
    y = np.nan_to_num(y.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    return y

def main(args):
    stim_out_path = Path('/om/user/imgriff/datasets/human_word_rec_SWC_2023/sounds')
    manifest = pd.read_pickle('/om2/user/imgriff/datasets/spatial_audio_pipeline/assets/human_attn_experiment_v00/full_eval_trial_manifest_new_fnames.pdpkl')

    with open("human_saddler_attn_expmt_cond_map.pkl", "rb") as f :
        # read condition dict from pickle
        cond_ix_dict = pickle.load(f)
    
    # drop columns with distractor in name
    unique_manifest = manifest[[col for col in manifest.columns if 'distractor' not in col]]
    # drop "gender_cond_td" column from manifest
    unique_manifest = unique_manifest.drop(columns=['gender_cond_td'])
    # drop duplicate rows from manifest
    unique_manifest = unique_manifest.drop_duplicates().reset_index(drop=True)


    fn_pkl_dst = '/om2/user/msaddler/spatial_audio_pipeline/assets/swc/manifest_all_words.pdpkl'
    word_dict = pickle.load(open("/om2/user/imgriff/datasets/commonvoice_9/en/cv_800_word_label_to_int_dict.pkl", 'rb'))
    words = list(word_dict.keys())
    words = [word.replace("'", "") for word in words]
    manifest_all_words = pd.read_pickle(fn_pkl_dst)
    # filter out words not in 'words' list
    manifest_all_words = manifest_all_words[manifest_all_words['word'].isin(words)]
    unique_manifest['word_int'] = unique_manifest['word'].map(word_dict)

    # distractor manifest - talkers & words not in test foreground set but still in vocab
    target_vocab = unique_manifest['word'].unique()
    distractor_manifest = manifest_all_words[~manifest_all_words['word'].isin(target_vocab)]
    # filter out target talkers
    distractor_manifest = distractor_manifest[~distractor_manifest.client_id.isin(unique_manifest.client_id.unique())]

    SR = 44100
    # timing in seconds 
    trial_dur = 4.5
    signal_dur = 2
    isi = 0.5
    win_dur = 10 # ms 
    mixture_onset = int((isi + signal_dur) * SR) # in frames

    # get bg conds
    cond_ix = args.array_id
    condition, snr = cond_ix_dict[cond_ix]

    path_to_bg_stim = Path("/om2/user/msaddler/spatial_audio_pipeline/assets/human_experiment_v00/")

    bg_stim = list((path_to_bg_stim / condition).glob('*.wav'))
    # sample unique manifest to get 180 males and 180 females but all 360 unique words 
    female_samps = unique_manifest[unique_manifest.gender == 'female'].sample(180)
    female_words = female_samps.word.unique()
    male_samps = unique_manifest[(unique_manifest.gender == "male") & ~unique_manifest.word.isin(female_words)].sample(180)
    wanted_manifest = pd.concat([female_samps, male_samps], axis=0, ignore_index=True)

    if "1-talker" in condition:
        f0_manifest = []

    cond_records = []
    for row in tqdm(wanted_manifest.itertuples()):
        record = {}
        trial_signal = np.zeros((int(SR * trial_dur)),dtype=np.float32)
        ix = row.Index # ix for background in 
        # are already cut/resampled/windowed
        cue_wav, cue_sr = librosa.load(row.cue_src_fn, sr=SR)
        target_wav, target_sr = librosa.load(row.src_fn, sr=SR)
        record['target_sr'] = target_sr
        record['cue_sr'] = cue_sr
        record['target_fn'] = row.src_fn
        record['cue_fn'] = row.cue_src_fn
        record['word'] = row.word
        record['word_int'] = row.word_int
        record['condition'] = condition
        record['snr'] = snr
        record['src_ix'] = row.src_ix
        record['client_id'] = row.client_id
        record['target_gender'] = row.gender

        # get talker f0 - bound to range of human voice 
        target_f0 = librosa.pyin(target_wav, fmin=80, fmax=300, sr=SR, fill_na=np.nan)
        avg_target_f0 = np.nanmean(target_f0)
        record['target_f0'] = avg_target_f0
        if "1-talker" in condition:
            distractor_gender = 'male' if ix < 180 else 'female'
            # get existing distractor signal used in binaural experiment
            distractor_eg = manifest[(manifest["src_ix"] == unique_manifest.src_ix[0]) & (manifest['distractor_gender'] == distractor_gender)].iloc[0]
            # is already cut/resampled/windowed
            bg_signal, _ = librosa.load(distractor_eg.distractor_src_fn, sr=SR)
            bg_talker_f0 = librosa.pyin(bg_signal, fmin=80, fmax=300, sr=SR, fill_na=np.nan)
            avg_bg_f0 = np.nanmean(bg_talker_f0)
            record['distractor_fn'] = distractor_eg.distractor_src_fn
            record['distractor_f0'] = avg_bg_f0
            record['distractor_gender'] = distractor_gender
         
        elif "4-talker" in condition:
            bg_talkers = distractor_manifest.sample(4).apply(lambda x: get_excerpt(x, dur=2, sr=44100, pad_with_context=True, jitter_fraction=0), axis=1).values
            bg_talkers = util_audio.set_dBSPL(np.stack(bg_talkers), 60) # set to 60 dB SPL
            bg_signal = bg_talkers.sum(axis=0) # set mixture to 60 dB

        elif "issn" in condition:
            bg_signal = util_audio.spectrally_matched_noise(target_wav, target_sr)
            if "festenplomp" in condition:
                donor_wav = distractor_manifest.sample(1).apply(lambda x: get_excerpt(x, dur=2,
                                                                                      sr=SR,
                                                                                      pad_with_context=True,
                                                                                      jitter_fraction=0), axis=1).values[0]
                bg_signal = util_audio.festen_plomp_fluctuating_noise(donor_wav, bg_signal, sr=SR, dbspl=60)
        else:
            ix  = ix % 360
            bg_signal, _ = librosa.load(bg_stim[ix], sr=SR)
            # crop and window to signal duration
            bg_signal = util_audio.ramp_hann(util_audio.pad_or_trim_to_len(bg_signal, int(2*SR)), win_dur, samplerate=SR)
        
        mixture = util_audio.combine_signal_and_noise(target_wav, bg_signal, snr)
        mixture = util_audio.ramp_hann(mixture, win_dur, samplerate=SR)

        mixture, mixture_scale_factor = rms_normalize_db(mixture, 60)
        # set cue to same level as target post scaling 
        cue_wav = util_audio.set_dBSPL(cue_wav, 60)
        cue_wav *= mixture_scale_factor

        # add cue to trial signal
        trial_signal[:len(cue_wav)] += cue_wav    
        trial_signal[mixture_onset:] += mixture
        trial_signal = util_audio.set_dBSPL(trial_signal, 60)
        # save trial signal
        # f string with digint padded to 3 places
        out_name = stim_out_path / f"condition_{cond_ix:02}"/ f'{row.word_int:03}.wav'
        # make out name directory if it doesn't exist
        out_name.parent.mkdir(parents=True, exist_ok=True)
        sf.write(out_name, trial_signal, samplerate=SR)
        record['mixture_fn'] = out_name
        cond_records.append(record)


    # make pd manifest from f0_manifest
    cond_manifest = pd.DataFrame.from_records(cond_records)
    # save f0_manifest as csv
    cond_manifest.to_pickle(stim_out_path / f"condition_{cond_ix:02}" / "manifest.pdpkl")

if __name__ == "__main__":
    # get array id from argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--array_id', type=int, required=True)
    args = parser.parse_args()
    main(args)

