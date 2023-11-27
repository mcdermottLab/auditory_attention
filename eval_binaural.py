import h5py
import numpy as np
import os
import pandas as pd
import pathlib
import pickle
import scipy.stats as stats
import soundfile as sf
import soxr
#! change below to spatial_attn_lighting if want to use with modular 
import src.spatial_attn_lightning as attn_tracking_lightning
import torch
import yaml

from argparse import ArgumentParser
from corpus.speaker_room_dataset import SpeakerRoomDataset
from tqdm.auto import tqdm

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

def spatialize(words, ir):
    """Uses pytorch to convolve all sounds in words with 2 channel IR given in ir"""
    n_words = words.shape[0]
    words_padded = [torch.nn.functional.pad(word, (ir.shape[0] - 1, 0)) for word in words]
    ir = ir.T.unsqueeze(1)
    words_padded = torch.stack(words_padded)
    spatialized = torch.nn.functional.conv1d(words_padded.view(n_words, 1, -1).cuda(), ir.cuda()).cuda()
    return spatialized


def run_eval(args):
    model_name = args.model_name
    checkpoint_path = args.ckpt_path
    cue_type = args.cue_type

    config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
    config['num_workers'] = args.n_jobs
    config['hparas']['batch_size'] = 30 # config['data']['loader']['batch_size'] // args.gpus
    config['noise_kwargs']['low_snr'] = 0
    config['noise_kwargs']['high_snr'] = 0

    #TODO handle multiple elevations
    idx = args.location_idx
    # re_run_mapping = pickle.load(open('/om2/user/rphess/Auditory-Attention/rerun_dict_3.pkl', 'rb'))
    loc_dict = pickle.load(open('/om2/user/rphess/Auditory-Attention/speaker_room_all_elev.pkl', 'rb'))
    idx += 2000
    target_loc = loc_dict[idx][0]
    distract_loc = loc_dict[idx][1]

    log_name = f"/{model_name}_{cue_type}_target_loc_{target_loc[0]}_{target_loc[1]}_distract_loc_{distract_loc[0]}_{distract_loc[1]}"
    print(log_name)

    experiment_dir = f"{args.exp_dir}/{model_name}"
    model = attn_tracking_lightning.BinauralAttentionModule.load_from_checkpoint(checkpoint_path=checkpoint_path, config=config).cuda()
    audio_transforms = model.audio_transforms
    # to inference mode 
    model = model.eval()

    new_room_manifest = pd.read_pickle('/om2/user/msaddler/spatial_audio_pipeline/assets/brir/mit_bldg46room1004/manifest_brir.pdpkl')
    only14_manifest = new_room_manifest[(new_room_manifest['src_dist'] == 1.4) & (new_room_manifest['index_room'] == 0)]
    ir_dict = dict()
    for loc in ['target', 'distractor', 'cue']:
        if loc == 'target':
            coords = target_loc
        elif loc == 'distractor':
            coords = distract_loc
        else:
            if cue_type == 'voice':
                coords = (0,0)
            else:
                coords = target_loc
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

    dataset = SpeakerRoomDataset('/om2/user/rphess/Auditory-Attention/final_binaural_manifest.pkl', '/om2/user/msaddler/spatial_audio_pipeline/assets/swc/manifest_all_words.pdpkl', cue_type)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=config['hparas']['batch_size'], shuffle=False, num_workers=config['num_workers'])

    output_dict = {'results': None, 'confusions': None}
    accuracies = []
    confusions = []

    with torch.no_grad(): 
        for batch in tqdm(dataloader):
            cue, fg, bg, label, confusion = batch

            cue = np.array(spatialize(cue.cuda(), cue_brir.cuda()).cpu()[:, :, 12500:137500])
            foreground = np.array(spatialize(fg.cuda(), tar_brir.cuda()).cpu()[:, :, 12500:137500])
            background = np.array(spatialize(bg.cuda(), dist_brir.cuda()).cpu()[:, :, 12500:137500])

            cue = audio_transforms(cue, None)[0]
            mixture = audio_transforms(foreground, background)[0]

            cue = cue.cuda()
            mixture = mixture.cuda()
            logits = model(cue, mixture, None)

            preds = logits.softmax(dim=-1).argmax(dim=-1).cpu().detach().numpy().astype('int')
            true_word = label.numpy().astype('int')
            con_word = confusion.numpy().astype('int')
            accuracy = (preds == true_word).astype('int')
            cons = (preds == con_word).astype('int')
            accuracies.append(accuracy)
            confusions.append(cons)
    accuracies = np.concatenate(accuracies)
    confusions = np.concatenate(confusions)
    output_dict['results'] = accuracies
    output_dict['confusions'] = confusions

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
        "--cue_type",
        default='voice',
        type=str,
        help="type of cue to use (Default: 'voice')",
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


    args = parser.parse_args()

    run_eval(args)


if __name__ == "__main__":
    cli_main()
