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
import h5py

def get_brir(azim=None, elev=None, coords=None, h5_fn=None, IR_df=None, out_sr=44_100):
    if coords is not None:
        azim, elev = coords
    df_row = IR_df[(IR_df['src_azim'] == azim) & (IR_df['src_elev'] == elev)]
    brir_ix = df_row['index_brir'].values[0]
    sr_src = df_row['sr'].values[0]
    with h5py.File(h5_fn, 'r') as f:
        brir = f['brir'][brir_ix]
    if out_sr != sr_src:
        brir = soxr.resample(brir.astype(np.float32), sr_src, out_sr)
    return brir

def main(args):

    ##########################
    # init experiment stimuli 
    ##########################
    expmt_dir = Path('/om/user/imgriff/datasets/human_azim_spotlight_SWC_2024/')
    manifest = pd.read_pickle(expmt_dir / "human_azim_spotlight_stim_manifest.pdpkl")

    with open(expmt_dir/"human_azim_spotlight_cond_map.pkl", "rb") as f :
        # read condition dict from pickle
        cond_ix_dict = pickle.load(f)
    
    with open(expmt_dir/"human_azim_spotlight_word_key.pkl", "rb") as f :
        word_key_dict = pickle.load(f)

    ###################
    # init audio params
    ###################

    SR = 44100
    dBSPL = 60
    # timing in seconds 
    trial_dur = 4.5
    signal_dur = 2
    isi = 0.5
    win_dur = 10 # ms 
    mixture_onset = int((isi + signal_dur) * SR) # in frames
    sig_len_frames = int(signal_dur * SR)

    ######################
    # Get job condition 
    ######################

    # get bg conds
    cond_ix = args.array_id
    target_azim, distractor_delta, distractor_azim = cond_ix_dict[cond_ix].values()
\
    print(f"Generating stimuli for condition {cond_ix}: {target_azim} target azim vs {distractor_azim} distractor azim")

    ################################
    # init brir and spatilization 
    ################################
    if args.anechoic:
        test_IR_manifest_dir = Path("/om2/user/msaddler/spatial_audio_pipeline/assets/brir/eval")
    else:
        test_IR_manifest_dir = Path("/om2/user/imgriff/spatial_audio_pipeline/assets/brir/mit_bldg46room1004_min_reverb")
    room_ix = 0
    test_IR_manifest_path = test_IR_manifest_dir / "manifest_brir.pdpkl"
    h5_fn = test_IR_manifest_dir / f"room{room_ix:04}.hdf5"
    new_room_manifest = pd.read_pickle(test_IR_manifest_path)
    only14_manifest = new_room_manifest[(new_room_manifest['index_room'] == room_ix)  & (new_room_manifest['src_dist'] == 1.4)]

    target_brir = get_brir(azim=target_azim, elev=0, h5_fn=h5_fn, IR_df=only14_manifest)
    distractor_brir = get_brir(azim=distractor_azim, elev=0, h5_fn=h5_fn, IR_df=only14_manifest)

    ###################
    # Init output dir 
    ###################
    if args.anechoic:
        out_dir = expmt_dir / "anechoic_room_sounds" / f"condition_{cond_ix:02}"
    else:
        out_dir = expmt_dir / "sounds" / f"condition_{cond_ix:02}"
    out_dir.mkdir(exist_ok=True, parents=True)
    
    ############################
    # sort stimuli manifets 
    ############################
    ## will write all possilbe stim combinations to disk - sample in prolific experiment logic 
    # sort manifest by word - orders output examples in 0-(Vocab - 1) in a-z order
    wanted_manifest = manifest.sort_values('word')

    ############################
    # generate stimuli
    ############################
    for row in tqdm(wanted_manifest.itertuples(), total=len(wanted_manifest)):
        trial_audio = np.zeros((int(SR * trial_dur), 2), dtype=np.float32) 
        cue_wav, _ = librosa.load(row.excerpt_cue_src_fn, sr=SR)
        target_wav, _ = librosa.load(row.excerpt_src_fn, sr=SR)
        distractor_wav, _ = librosa.load(row.excerpt_distractor_src_fn, sr=SR)

        # set levels for each signal 
        cue_wav = util_audio.set_dBSPL(cue_wav, dBSPL)
        target_wav = util_audio.set_dBSPL(target_wav, dBSPL)
        distractor_wav = util_audio.set_dBSPL(distractor_wav, dBSPL)

        # spatialize 
        cue_spatial = util_audio.spatialize_sound(cue_wav, target_brir)
        target_spatial = util_audio.spatialize_sound(target_wav, target_brir)
        distractor_spatial = util_audio.spatialize_sound(distractor_wav, distractor_brir)

        # combine target and distractor
        mixture = target_spatial + distractor_spatial

        # add to trial audio_array 
        trial_audio[:len(cue_spatial), :] = cue_spatial
        trial_audio[mixture_onset:, :] = mixture

        # setup naming and write
        targ_sex_initial = row.gender[0]
        exp_word_ix = row.exp_word_ix
        out_name = out_dir / f'{targ_sex_initial}_{exp_word_ix:03}.wav'
        # write audio 
        sf.write(out_name, trial_audio.astype('float32'), samplerate=SR)
            

if __name__ == "__main__":
    # get array id from argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--array_id', type=int, required=True)
    parser.add_argument('--anechoic', action='store_true')
    args = parser.parse_args()
    main(args)

