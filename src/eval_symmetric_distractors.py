import h5py
import numpy as np
import os
import pandas as pd
import pathlib
from pathlib import Path
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
from corpus.speech_and_texture_test import SpeechAndTextureTestSet
from corpus.binaural_swc_currated_pd import SWCHumanExperimentStimDataset
from tqdm.auto import tqdm
from datetime import datetime
import sys


torch.set_float32_matmul_precision('medium') # use same as training
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

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
        # spatialized = spatialized[:, :, self.start_frame:self.end_frame]
        return spatialized

def run_eval(args):
    # seed rngs 
    torch.manual_seed(args.location_idx)
    np.random.seed(args.location_idx)
    
    checkpoint_path = args.ckpt_path
    cue_type = args.cue_type
    model_name = args.model_name if args.model_name != '' else Path(args.config).stem

    # load model config 
    config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
    config['num_workers'] = args.n_jobs
    config['hparas']['batch_size'] = 30 # config['data']['loader']['batch_size'] // args.gpus
    # get model input sr for brir resampling
    model_in_sr = config['audio']['rep_kwargs']['sr']

    dual_task_arch =  config['model'].get("cue_loc_task", False)


    idx = args.location_idx
    test_dict = pickle.load(open(args.test_manifest, 'rb'))
    n_per_job = args.n_per_job
    start = idx * n_per_job
    end = start + n_per_job

    experiment_dir = f"{args.exp_dir}/{model_name}"
    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir, exist_ok=True)

    model = attn_tracking_lightning.BinauralAttentionModule.load_from_checkpoint(checkpoint_path=checkpoint_path, config=config, strict=True).cuda()
    # define audio transforms to standardize eval transforms across models (v05 skips BinauralCombineWithRandomDBSNR)
    audio_transforms_0_db = at.AudioCompose([
                        at.AudioToTensor(),
                        at.BinauralCombineWithRandomDBSNR(low_snr=0,    # is 0 dB
                                                          high_snr=0,
                                                          v2_demean=True), # is 0 dB 
                        at.BinauralRMSNormalizeForegroundAndBackground(rms_level=0.02,
                                                                       v2_demean=True), # 20 * np.log10(0.02/20e-6) = 60 dB SPL 
                ])

    audio_transforms_0_db = audio_transforms_0_db.cuda()
    # to inference mode 
    model = model.eval()
    coch_gram = model.coch_gram.cuda()

    # set up dataset and dataloader
    if args.modulated_ssn_distractors:
        print("Using modulated ssn distractors")
    
    if args.ssn_distractors:
        print("Using ssn distractors")
    

    
    # use anechoic BRIRs for testing
    new_room_manifest = None 
    only14_manifest = None
    
    for idx in range(start,end):
        target_loc = test_dict[idx]['target_loc']
        distract_loc = test_dict[idx]['distract_loc']
        threshold_snr = test_dict[idx]['snr']
        with_noise = test_dict[idx].get('with_noise', False)
        with_textures = test_dict[idx].get('with_textures', False)
        print(test_dict[idx])

        symmetric_distractor = args.run_1_distractor or test_dict[idx].get('symmetric_distractor', False)
        sym_dist_str = 'symmetric distractors' if symmetric_distractor else 'single distractor'
        print(f"Running evaluation with {sym_dist_str}")
        if args.sim_human_array_exmpt:
            dataset = SWCHumanExperimentStimDataset(path='/om/user/imgriff/datasets/human_word_rec_SWC_2024/full_cue_target_distractor_df_w_meta.pdpkl',
                                                    run_all_stim=args.run_all_stim,
                                                    ssn_distractor=args.ssn_distractors,
                                                    sr=model_in_sr)
        elif args.texture_distractor or with_textures:
            print("Using textures as distractors")
            dataset = SpeechAndTextureTestSet(file_path='/om/user/imgriff/datasets/speech_in_synthetic_textures/separated_sources/stim.hdf5',
                                            separated_signals=True,
                                            symmetric_distractor=True)
        else:
            dataset = SpeakerRoomDataset(manifest_path='/om2/user/rphess/Auditory-Attention/final_binaural_manifest.pkl',
                                        excerpt_path='/om2/user/msaddler/spatial_audio_pipeline/assets/swc/manifest_all_words.pdpkl',
                                        cue_type=cue_type,
                                        sr=model_in_sr,
                                        symmetric_distractor_test=True,
                                        modulated_ssn_distractors=args.modulated_ssn_distractors,
                                        ssn_distractors=args.ssn_distractors,
                                        return_stim_ixs=True) 
            
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=config['hparas']['batch_size'], shuffle=False, num_workers=config['num_workers'])
        
        if test_dict[idx].get('test_room_meta', False):
            test_manifest_path = test_dict[idx]['test_room_meta']['room_manifest']
            test_room_idx = test_dict[idx]['test_room_meta']['index_room']
            new_room_manifest = pd.read_pickle(test_manifest_path)
            only14_manifest = new_room_manifest[(new_room_manifest['index_room'] == test_room_idx)  & (new_room_manifest['src_dist'] == 1.4)]
            h5_fn = test_dict[idx]['test_room_meta']['h5_fn']
            if 'eval' in Path(test_manifest_path).as_posix():
                room_str = f'eval_room{test_room_idx:04}'
            elif 'min_reverb' in Path(test_manifest_path).as_posix():
                room_str = f'min_reverb_room{test_room_idx:04}'
            else:
                room_str = f'mitb46_room{test_room_idx:04}'


        elif new_room_manifest == None:
            # use anechoic BRIRs for testing
            new_room_manifest = pd.read_pickle('/om2/user/msaddler/spatial_audio_pipeline/assets/brir/ruggles_shinncunningham_2011/manifest_brir.pdpkl')
            h5_fn = f'/om2/user/msaddler/spatial_audio_pipeline/assets/brir/ruggles_shinncunningham_2011/room0000.hdf5'
            only14_manifest = new_room_manifest[(new_room_manifest['index_room'] == 0)]
            room_str = f'ruggles_shinncunningham_2011_room0000'


        audio_transforms_test_db = at.AudioCompose([
                        at.AudioToTensor(),
                        at.BinauralCombineWithRandomDBSNR(low_snr=threshold_snr,    
                                                          high_snr=threshold_snr,
                                                          v2_demean=True), 
                        at.BinauralRMSNormalizeForegroundAndBackground(rms_level=0.02,
                                                                       v2_demean=True)
                ])
        audio_transforms_test_db = audio_transforms_test_db.cuda()

        # Add modifications to log name based on flags for test conditions 
        sym_str = "" if symmetric_distractor else "_1_distractor"
        if args.texture_distractor or with_textures:
            dist_str = "texture_distractor"
        elif args.ssn_distractors:
            dist_str = "ssn_distractor"
        elif args.modulated_ssn_distractors:
            dist_str = "modulated_ssn_distractor"
        elif args.noise_distractor or with_noise:
            dist_str = "whitenoise_distractor"
        else:
            dist_str = "speech_distractor"

        all_stim_str = 'all_stim' if args.run_all_stim else 'subset_stim'
        log_name = f"/{model_name}_cue_{cue_type}_target_loc_{target_loc[0]}_{target_loc[1]}_distract_loc_{distract_loc[0]}_{distract_loc[1]}_{int(threshold_snr)}_SNR_{dist_str}_{room_str}_{sym_str}_{all_stim_str}"  
        # replace __ with _ in log_name in case format produces it 
        log_name = log_name.replace('__', '_')      
        print(log_name)
        output_name = str(experiment_dir) + log_name + '.pkl'
        print("Overwrite ", args.overwrite)
        if not args.overwrite and os.path.exists(output_name):
            continue
        
        ir_dict = dict()
        for loc in ['target', 'distractor_l', 'distractor_r']:
            if loc == 'target':
                coords = target_loc
            elif loc == 'distractor_r':
                coords = distract_loc.copy()
                coords[0] = 360 - coords[0] if coords[0] != 0 else 0 
            elif loc == 'distractor_l':
                coords = distract_loc.copy()
            df_row = only14_manifest[(only14_manifest['src_azim'] == coords[0]) & (only14_manifest['src_elev'] == coords[1])]
            index_brir = df_row['index_brir'].values[0]
            sr_src = df_row['sr'].values[0]
            with h5py.File(h5_fn, 'r') as f:
                brir = f['brir'][index_brir]
            if model_in_sr != sr_src:
                brir = soxr.resample(brir.astype(np.float32), sr_src, model_in_sr)
            ir_dict[loc] = brir.astype(np.float32)

        tar_brir = Spatialize(ir_dict['target'], model_sr=model_in_sr).cuda()
        dist_brir_l = Spatialize(ir_dict['distractor_l'], model_sr=model_in_sr).cuda()
        dist_brir_r = Spatialize(ir_dict['distractor_r'], model_sr=model_in_sr).cuda()

        output_dict = {'results': None, 'confusions': None}
        accuracies = []
        confusions = []
        pred_list = []
        true_word_int = []
        stim_ix_list = []
        if args.texture_distractor or with_textures:
            texture_list = []

        with torch.no_grad(): 
            for batch in tqdm(dataloader):
                if args.texture_distractor or with_textures:
                    cue, fg, bg, bg_2, label, texture, stim_ixs = batch
                    dist_word_label, dist_word_label2 = None, None 
                else:
                    cue, fg, bg, bg_2, label, dist_word_label, dist_word_label2, stim_ixs = batch
                stim_ix_list.append(stim_ixs)
                # make random noise distractors
                if args.noise_distractor:
                    bg = torch.randn_like(bg)
                    bg_2 = torch.randn_like(bg_2)
                # spatialize signals 
                cue = tar_brir(cue.cuda())
                foreground = tar_brir(fg.cuda())
                background_l = dist_brir_l(bg.cuda())
                if not symmetric_distractor:
                    background_r = None
                else:
                    background_r = dist_brir_r(bg_2.cuda())
                ## set to desired SNR and SPL 
                cue, _ = audio_transforms_0_db(cue, None)
                # Set left/right distractor to same level and mix
                mixed_bg, _ = audio_transforms_0_db(background_l, background_r)
                # Set foreground to distractor at given SNR, then set to 60dB  
                mixture, _ = audio_transforms_test_db(foreground, mixed_bg)

                cue, mixture = coch_gram(cue, mixture)
                logits = model(cue, mixture, None)
                if dual_task_arch:
                    logits, _ = logits # unpack dual task output (word, location)
                # Unpack desired metrics 
                preds = logits.softmax(dim=-1).argmax(dim=-1).cpu().detach().numpy().astype('int')
                true_word = label.numpy().astype('int')
                accuracy = (preds == true_word).astype('int')
                accuracies.append(accuracy)
                pred_list.append(preds)
                true_word_int.append(true_word)
                if args.texture_distractor or with_textures:
                    texture_list.append(texture)
                else:
                    dist_word_label = dist_word_label.numpy().astype('int')
                    dist_word_label2 = dist_word_label2.numpy().astype('int')
                    # confusion made if preds == dist_word_label or dist_word_label2
                    cons_1 = (preds == dist_word_label).astype('int')
                    cons_2 = (preds == dist_word_label2).astype('int')
                    cons = np.bitwise_or(cons_1, cons_2) # get union of confusions
                    confusions.append(cons)
        accuracies = np.concatenate(accuracies)
        preds = np.concatenate(pred_list)
        true_word_int = np.concatenate(true_word_int)
        stim_ix_list = np.concatenate(stim_ix_list)
        output_dict['results'] = accuracies
        output_dict['preds'] = preds
        output_dict['true_word_int'] = true_word_int
        output_dict['stim_ix_list'] = stim_ix_list
        if args.texture_distractor or with_textures:
            texture_list = np.concatenate(texture_list)
            output_dict['textures'] = texture_list
        else:
            confusions = np.concatenate(confusions)
            output_dict['confusions'] = confusions

        print(log_name)    
        print(f"Test {distract_loc[0]} azimuth distractor at {threshold_snr} SNR in {room_str}")
        print(f"Accuracy: {accuracies.mean()}")
        if not args.texture_distractor:
            print(f"Confusions: {confusions.mean()}")

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
        "--test_manifest",
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
        default='',
        type=str,
        help="Name of model to use in file name.",
    )
    parser.add_argument(
        "--location_idx",
        type=int,
        help="index into saved location dictionary",
    )
    parser.add_argument(
        "--n_per_job",
        default=10,
        type=int,
        help="Number location conditions to run per job. (Default: 10)",
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
    # create overwrite flag to handle overwrite of existing results
    parser.add_argument(
        "--run_1_distractor",
        action=argparse.BooleanOptionalAction,
        help="If true, will run just 1 distractor",
    )
    parser.add_argument(
        "--ssn_distractors",
        action=argparse.BooleanOptionalAction,
        help="If true, will use standard ssn maskers as distractors"
    )
    parser.add_argument(
        "--modulated_ssn_distractors",
        action=argparse.BooleanOptionalAction,
        help="If true, will use festen and plomp style modulated ssn maskers as distractors"
    )
    parser.add_argument(
        "--sim_human_array_exmpt",
        action=argparse.BooleanOptionalAction,
        help="If true, will use dataset to support conditions simulating human speaker array experiment."
    )
    parser.add_argument(
        "--run_all_stim",
        action=argparse.BooleanOptionalAction,
        help="If true, will run all stimuli in the dataset."
    )
    parser.add_argument(
        "--noise_distractor",
        action=argparse.BooleanOptionalAction,
        help="If true, will use noise distractors instead of speech distractors."
    )
    parser.add_argument(
        "--texture_distractor",
        action=argparse.BooleanOptionalAction,
        help="If true, will use textures as distractors instead of speech distractors."
    )

    args = parser.parse_args()

    run_eval(args)


if __name__ == "__main__":
    cli_main()
