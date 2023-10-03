# Following is originally copied from PyTorch RNN-T ASR Example:
# https://github.com/pytorch/audio/tree/820b383b3b21fc06e91631a5b1e6ea1557836216/examples/asr/librispeech_emformer_rnnt

import h5py
import numpy as np
import os
import pandas as pd
import pathlib
import pickle
import scipy.stats as stats
import soundfile as sf
import soxr
import torch
import yaml

from argparse import ArgumentParser
from pytorch_lightning import Trainer, seed_everything

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
seed_everything(1)

def mass_spatialize(words, ir):
    """Uses pytorch to convolve all sounds in words with 2 channel IR given in ir"""
    n_words = words.shape[0]
    words_padded = [torch.nn.functional.pad(word, (ir.shape[0] - 1, 0)) for word in words]
    ir = ir.T.unsqueeze(1)
    words_padded = torch.stack(words_padded)
    spatialized = torch.nn.functional.conv1d(words_padded.view(n_words, 1, -1).cuda(), ir.cuda()).cuda()
    return spatialized


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
    y = pad_or_trim_to_len(y, int(dur * sr))
    y = np.nan_to_num(y.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    return y


def run_eval(args):
    model_name = args.model_name
    checkpoint_path = args.ckpt_path

    config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
    config['num_workers'] = args.n_jobs
    config['hparas']['batch_size'] = 1 # config['data']['loader']['batch_size'] // args.gpus
    config['noise_kwargs']['low_snr'] = args.snr
    config['noise_kwargs']['high_snr'] = args.snr
    config['corpus']['cue_type'] = args.cue_type
    idx = args.location_idx
    # re_run_mapping = pickle.load(open('/om2/user/rphess/Auditory-Attention/rerun_dict_3.pkl', 'rb'))
    loc_dict = pickle.load(open('/om2/user/rphess/Auditory-Attention/speaker_room_0_elev_conditions.pkl', 'rb'))
    target_loc = loc_dict[idx][0]
    distract_loc = loc_dict[idx][1]

    log_name = f"/bin_attn_task_target_loc_{target_loc[0]}_{target_loc[1]}_distract_loc_{distract_loc[0]}_{distract_loc[1]}"
    print(log_name)

    experiment_dir = pathlib.Path(args.exp_dir) / f"test_{args.cue_type}_{model_name}_{args.snr}dB"

    # Get model module dynamically
    # If using config that specifies architecture, use spatial_attn_lightning
    # TO DO: clean up and make one module for both
    if 'kernel' in config['model']:
        from src.spatial_attn_lightning import BinauralAttentionModule 
    else:
        from src.binaural_attn_lightning import BinauralAttentionModule
    

    model = BinauralAttentionModule.load_from_checkpoint(checkpoint_path=checkpoint_path, config=config).cuda()
    audio_transforms = model.audio_transforms

    new_room_manifest = pd.read_pickle('/om2/user/msaddler/spatial_audio_pipeline/assets/brir/mit_bldg46room1004/manifest_brir.pdpkl')
    only14_manifest = new_room_manifest[new_room_manifest['src_dist'] == 1.4]
    ir_dict = dict()
    for loc in ['target', 'distractor', 'cue']:
        if loc == 'target':
            coords = target_loc
        elif loc == 'distractor':
            coords = distract_loc
        elif loc == 'cue':
            coords = [0, 0]
        df_row = only14_manifest[(only14_manifest['src_azim'] == coords[0]) & (only14_manifest['src_elev'] == coords[1])]
        h5_fn = f'/om2/user/msaddler/spatial_audio_pipeline/assets/brir/mit_bldg46room1004/room000{df_row["index_room"].values[0]}.hdf5'
        index_brir = df_row['index_brir'].values[0]
        sr_src = df_row['sr'].values[0]
        with h5py.File(h5_fn, 'r') as f:
            brir = f['brir'][index_brir]
        sr = 50000
        brir = soxr.resample(brir.astype(np.float32), sr_src, sr)
        ir_dict[loc] = brir

    tar_brir = torch.from_numpy(ir_dict['target'])
    tar_brir = torch.flip(tar_brir, dims=[0])
    dist_brir = torch.from_numpy(ir_dict['distractor'])
    dist_brir = torch.flip(dist_brir, dims=[0])
    cue_brir = torch.from_numpy(ir_dict['cue'])
    cue_brir = torch.flip(cue_brir, dims=[0])

    class_map = pickle.load( open("/om2/user/imgriff/datasets/commonvoice_9/en/cv_800_word_label_to_int_dict.pkl", "rb" )) 
    class_map = {k.replace("'", ''):v for k,v in class_map.items()}

    fn_pkl_dst = '/om2/user/msaddler/spatial_audio_pipeline/assets/swc/manifest_all_words.pdpkl'
    all_words_not_filtered = pd.read_pickle(fn_pkl_dst)
    df = pd.read_pickle('/om2/user/rphess/Auditory-Attention/binaural_test_manifest.pdpkl')
    foregrounds = []
    cues = []
    for i, row in df.iterrows():
        src_ix = row['src_ix']
        cue_ix = row['cue_src_ix']
        src_row = all_words_not_filtered.iloc[src_ix]
        cue_row = all_words_not_filtered.iloc[cue_ix]
        foregrounds.append(get_excerpt(src_row))
        cues.append(get_excerpt(cue_row))
    df['loaded_foreground'] = foregrounds
    df['loaded_cue'] = cues

    male_df = df[df['gender'] == 'male']
    female_df = df[df['gender'] == 'female']

    output_dict = {'m_m': {'target_loc': target_loc, 'distract_loc': distract_loc, 'results': None, 'confusions': None}, 
                   'm_f': {'target_loc': target_loc, 'distract_loc': distract_loc, 'results': None, 'confusions': None}, 
                   'f_m': {'target_loc': target_loc, 'distract_loc': distract_loc, 'results': None, 'confusions': None}, 
                   'f_f': {'target_loc': target_loc, 'distract_loc': distract_loc, 'results': None, 'confusions': None},
                   }
    for condition in ['m_m', 'm_f', 'f_m', 'f_f']:
        results = []
        confusions = []
        if condition[0] == 'm':
            tar_df = male_df
        else:
            tar_df = female_df
        if condition[2] == 'm':
            dist_df = male_df
        else:
            dist_df = female_df
        for i, row in tar_df.iterrows():
            cue = torch.from_numpy(row['loaded_cue']).unsqueeze(0)
            fg = torch.from_numpy(row['loaded_foreground']).unsqueeze(0)
            client_id = row['client_id']
            word = row['word']
            label = class_map[word]
            distractor = dist_df[dist_df['client_id'] != client_id].sample(1)
            distractor_signal = torch.from_numpy(distractor['loaded_foreground'].values[0]).unsqueeze(0)
            distractor_label = class_map[distractor['word'].values[0]]

            cue = mass_spatialize(cue.cuda(), cue_brir.cuda()).cpu()
            cue = np.array(cue[:, :, 12500:137500])
            #! Force cue to batch, channel, time format
            cue = audio_transforms(cue, None)[0].view(-1, 2, 125000)

            fg = mass_spatialize(fg.cuda(), tar_brir.cuda()).cpu()
            fg = np.array(fg[:, :, 12500:137500])
            bg = mass_spatialize(distractor_signal.cuda(), dist_brir.cuda()).cpu()
            bg = np.array(bg[:, :, 12500:137500])
            #! Force scene to batch, channel, time format
            scene = audio_transforms(fg, bg)[0].view(-1, 2, 125000)

            out = model.forward(cue.cuda(), scene.cuda(), cue_mask_ixs=None)
            softmax_outputs = torch.nn.functional.softmax(out, dim=-1)
            result = int(torch.argmax(softmax_outputs, dim=-1).cpu())
            results.append(label == result)
            confusions.append(distractor_label == result)
        res_err = stats.sem(results)
        res = np.mean(results)
        con_err = stats.sem(confusions)
        con = np.mean(confusions)
        output_dict[condition]['results'] = (res, res_err)
        output_dict[condition]['confusions'] = (con, con_err)

    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)
    with open(str(experiment_dir) + log_name + '.pkl', 'wb') as f:
        pickle.dump(output_dict, f)

def cli_main():
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default="", help='Path to model config.')
    parser.add_argument(
        "--exp_dir",
        default=pathlib.Path("./exp"),
        type=pathlib.Path,
        help="Directory to save checkpoints and logs to. (Default: './exp')",
    )
    parser.add_argument(
        "--ckpt_path",
        default=pathlib.Path("./exp"),
        type=pathlib.Path,
        help="path to checkpoint (Default: './exp')",
    )
    parser.add_argument(
        "--model_name",
        default='BinauralAttn_Word_Task_Voice_Cue',
        type=str,
        help="Name of model to use in file name.",
    )
    parser.add_argument(
        "--location_idx",
        type=int,
        help="index into saved location dictionary",
    )
    parser.add_argument(
        "--num_nodes",
        default=1,
        type=int,
        help="Number of nodes to use for training. (Default: 1)",
    )
    parser.add_argument(
        "--gpus",
        default=1,
        type=int,
        help="Number of GPUs per node to use for training. (Default: 1)",
    )
    parser.add_argument(
        "--n_jobs",
        default=0,
        type=int,
        help="Number of CPUs for dataloader. (Default: 0)",
    )
    parser.add_argument(
        "--snr",
        default=0,
        type=int,
        help="Number of CPUs for dataloader. (Default: 0)",
    )
    parser.add_argument(
        "--cue_type",
        default='voice_and_location',
        type=str,
        help="Type of cue to use in evaluation. One of `voice_and_location`, `voice`, `location` (Default: `voice_and_location`)",
    )


    args = parser.parse_args()

    run_eval(args)


if __name__ == "__main__":
    cli_main()
