import h5py
import numpy as np
import os
import pandas as pd
import pathlib
import pickle
import scipy.stats as stats
import soxr
#! change below to spatial_attn_lighting if want to use with modular 
import src.spatial_attn_lightning as attn_tracking_lightning
import src.audio_transforms as at
import torch
import yaml

import argparse
from argparse import ArgumentParser
from corpus.speaker_room_dataset import SpeakerRoomDataset
from tqdm.auto import tqdm
from datetime import datetime

torch.set_float32_matmul_precision('medium') # use same as training
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

# make torch.nn.Module version of spatilaize
class Spatialize(torch.nn.Module):
    def __init__(self, ir, model_sr=50_000):
        super(Spatialize, self).__init__()
        ir = torch.flip(torch.from_numpy(ir), dims=[0]).float()
        self.n_taps = ir.shape[0]
        ir = ir.T.unsqueeze(1)
        # set center crop of 2.5 seconds relative to model_sr
        self.start_frame = int(model_sr * 0.25)
        self.end_frame = int(model_sr * 2.75)

        self.register_buffer("ir", ir)

    def forward(self, words):
        n_words = words.shape[0]
        # pad last dim of words with ir.shape[0] - 1 zeros
        words_padded = torch.nn.functional.pad(words, (self.n_taps - 1, 0))
        spatialized = torch.nn.functional.conv1d(words_padded.view(n_words, 1, -1), self.ir)
        # resize to desired shape
        spatialized = spatialized[:, :, self.start_frame:self.end_frame]
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
    # get model input sr for brir resampling
    model_in_sr = config['audio']['rep_kwargs']['sr']

    #TODO handle multiple elevations
    idx = args.location_idx
    # re_run_mapping = pickle.load(open('/om2/user/rphess/Auditory-Attention/rerun_dict_3.pkl', 'rb'))
    # loc_dict = pickle.load(open('/om2/user/rphess/Auditory-Attention/speaker_room_all_elev.pkl', 'rb'))
    loc_dict = pickle.load(open(args.location_manifest, 'rb'))

    n_per_job = 10
    start = idx * n_per_job
    end = start + n_per_job

    experiment_dir = f"{args.exp_dir}/{model_name}"
    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)
    model = attn_tracking_lightning.BinauralAttentionModule.load_from_checkpoint(checkpoint_path=checkpoint_path, config=config, strict=False).cuda()
    # define audio transforms to standardize eval transforms across models (v05 skips BinauralCombineWithRandomDBSNR)
    audio_transforms = at.AudioCompose([
                    at.AudioToTensor(),
                    at.BinauralCombineWithRandomDBSNR(low_snr=config['noise_kwargs']['low_snr'],    # is 0 dB
                                                      high_snr=config['noise_kwargs']['high_snr']), # is 0 dB 
                    at.BinauralRMSNormalizeForegroundAndBackground(rms_level=0.02), # 20 * np.log10(0.02/20e-6) = 60 dB SPL 
            ])
    audio_transforms = audio_transforms.cuda()
    # to inference mode 
    model = model.eval()
    coch_gram = model.coch_gram.cuda()

    # set up dataset and dataloader
    dataset = SpeakerRoomDataset(manifest_path='/om2/user/rphess/Auditory-Attention/final_binaural_manifest.pkl',
                                excerpt_path='/om2/user/msaddler/spatial_audio_pipeline/assets/swc/manifest_all_words.pdpkl',
                                cue_type=cue_type,
                                sr=model_in_sr) 
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=config['hparas']['batch_size'], shuffle=False, num_workers=config['num_workers'])

    new_room_manifest = pd.read_pickle('/om2/user/msaddler/spatial_audio_pipeline/assets/brir/mit_bldg46room1004/manifest_brir.pdpkl')
    ## Get room ix based on test
    if 'front_back' in experiment_dir:
        room_ix = 1 # 1 is room rotated for compat with human front-back experiment
    else:
        room_ix = 0
    only14_manifest = new_room_manifest[(new_room_manifest['src_dist'] == 1.4) & (new_room_manifest['index_room'] == room_ix)]
    for idx in range(start,end):
        target_loc = loc_dict[idx][0]
        distract_loc = loc_dict[idx][1]

        log_name = f"/{model_name}_cue_{cue_type}_target_loc_{target_loc[0]}_{target_loc[1]}_distract_loc_{distract_loc[0]}_{distract_loc[1]}"        
        print(log_name)
        output_name = str(experiment_dir) + log_name + '.pkl'
        if idx % 10 == 0:
            print("Overwrite ", args.overwrite)
        if not args.overwrite and os.path.exists(output_name):
            continue
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
            h5_fn = f'/om2/user/msaddler/spatial_audio_pipeline/assets/brir/mit_bldg46room1004/room000{room_ix}.hdf5'
            index_brir = df_row['index_brir'].values[0]
            sr_src = df_row['sr'].values[0]
            with h5py.File(h5_fn, 'r') as f:
                brir = f['brir'][index_brir]
            if model_in_sr != sr_src:
                brir = soxr.resample(brir.astype(np.float32), sr_src, model_in_sr)
            ir_dict[loc] = brir.astype(np.float32)

        tar_brir = Spatialize(ir_dict['target'], model_sr=model_in_sr).cuda()
        dist_brir = Spatialize(ir_dict['distractor'], model_sr=model_in_sr).cuda()
        cue_brir = Spatialize(ir_dict['cue'], model_sr=model_in_sr).cuda()

        output_dict = {'results': None, 'confusions': None}
        accuracies = []
        confusions = []
        pred_list = []
        true_word_int = []

        with torch.no_grad(): 
            for batch in tqdm(dataloader):
                cue, fg, bg, label, confusion = batch

                cue = cue_brir(cue.cuda())
                foreground = tar_brir(fg.cuda())
                background = dist_brir(bg.cuda())

                cue = audio_transforms(cue, None)[0]
                mixture = audio_transforms(foreground, background)[0]
                # cue = cue.cuda()
                # mixture = mixture.cuda()
                cue, mixture = coch_gram(cue, mixture)
                logits = model(cue, mixture, None)

                preds = logits.softmax(dim=-1).argmax(dim=-1).cpu().detach().numpy().astype('int')
                true_word = label.numpy().astype('int')
                con_word = confusion.numpy().astype('int')
                accuracy = (preds == true_word).astype('int')
                cons = (preds == con_word).astype('int')
                accuracies.append(accuracy)
                confusions.append(cons)
                pred_list.append(preds)
                true_word_int.append(true_word)
        accuracies = np.concatenate(accuracies)
        confusions = np.concatenate(confusions)
        preds = np.concatenate(pred_list)
        true_word_int = np.concatenate(true_word_int)

        output_dict['results'] = accuracies
        output_dict['confusions'] = confusions
        output_dict['preds'] = preds
        output_dict['true_word_int'] = true_word_int


        with open(output_name, 'wb') as f:
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
        "--location_manifest",
        default=pathlib.Path("/om2/user/imgriff/Auditory-Attention/speaker_room_0_elev_conditions.pkl"),
        type=pathlib.Path,
        help="path manifest of target and distractor locations to use for evaluation",
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
    # create overwrite flag to handle overwrite of existing results
    parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        help="If true, will overwrite existing results",
    )

    args = parser.parse_args()

    run_eval(args)


if __name__ == "__main__":
    cli_main()
