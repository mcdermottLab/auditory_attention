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
from corpus.speech_and_texture_test import SpeechAndTextureTestSet
from corpus.binaural_swc_currated_pd import SWCHumanExperimentStimDataset
from tqdm.auto import tqdm
from datetime import datetime
import sys


torch.set_float32_matmul_precision('medium') # use same as training
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

def db_to_rms(db):
    return np.power(10,(db/20)) * 20e-6

class SpatializeAllLocs(torch.nn.Module):
    def __init__(self, ir_array, model_sr=44_100):
        super(SpatializeAllLocs, self).__init__()

        self.n_locs, self.n_taps, self.n_channels = ir_array.shape
        print(f"Loaded {self.n_locs} BRIRs with {self.n_taps} taps and {self.n_channels} channels")
        ## reshape all brirs
        ir_array = ir_array.reshape(self.n_channels, 1, self.n_locs, self.n_taps)
        print(ir_array.shape)
        brir_kernel = torch.flip(torch.from_numpy(ir_array).float(), dims=[-1])
        self.register_buffer("brir_kernel", brir_kernel)
        # self.start_frame = int(model_sr * 0.25)
        # self.end_frame = int(model_sr * 2.75)

    def forward(self, signal):
        n = signal.shape[0]
        padded_texture = torch.nn.functional.pad(signal, (self.n_taps - 1, 0)).view(n, 1, 1, -1)
        # repeat texture along height dimension for conv 
        padded_texture = padded_texture.repeat(1, 1, self.n_locs, 1)

        spatialized = torch.nn.functional.conv2d(padded_texture,
                                                self.brir_kernel,
                                                stride=1)  
        spatialized = spatialized.squeeze() 

        return spatialized

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
        # resize to desired shape - so we can add texture 
        spatialized = spatialized[:, :, self.start_frame:self.end_frame]
        return spatialized


def get_texture_eg(texture_dataset):
    ix = np.random.randint(len(texture_dataset))
    _, _, texture_signal, _, texture_label = texture_dataset[ix]

    # texture_labels.append(texture_label)
    texture_signal = torch.from_numpy(texture_signal).unsqueeze(0)
    return texture_signal, texture_label

def gen_pink_noise_batch(duration, n_examples=1):
    uneven = duration % 2
    X = torch.randn((n_examples, duration // 2 + 1 + uneven)) + 1j * torch.randn((n_examples, duration // 2 + 1 + uneven))
    S = torch.sqrt(torch.arange(X.shape[1]) + 1.)  # +1 to avoid divide by zero
    Y = (torch.fft.irfft(X / S)).real
    if uneven:
        Y = Y[:, : -1]
    return Y

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
    config['hparas']['batch_size'] = 32 # config['data']['loader']['batch_size'] // args.gpus
    # get model input sr for brir resampling
    model_in_sr = config['audio']['rep_kwargs']['sr']

    dual_task_arch =  config['model'].get("cue_loc_task", False)


    if 'backbone' in model_name:
        if args.backbone_with_ecdf_gains:
            config['model']['backbone_with_ecdf_gains'] = True

    idx = args.location_idx
    test_dict = pickle.load(open(args.test_manifest, 'rb'))
    n_per_job = args.n_per_job
    start = idx * n_per_job
    end = start + n_per_job

    experiment_dir = f"{args.exp_dir}/{model_name}"
    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir, exist_ok=True)

    model = attn_tracking_lightning.BinauralAttentionModule.load_from_checkpoint(checkpoint_path=checkpoint_path,
                                                                                 config=config,
                                                                                 strict=False if args.backbone_with_ecdf_gains else True).cuda()

    # to inference mode 
    model = model.eval()
    coch_gram = model.coch_gram.cuda()

    
    dataset = SWCHumanExperimentStimDataset(path='/om/user/imgriff/datasets/human_word_rec_SWC_2024/full_cue_target_distractor_df_w_meta.pdpkl',
                                            run_all_stim=args.run_all_stim,
                                            sr=model_in_sr)
    if args.texture_distractor:
        print("Using textures as background noise")
        texture_dataset = SpeechAndTextureTestSet(file_path='/om/user/imgriff/datasets/speech_in_synthetic_textures/separated_sources/stim.hdf5',
                                            separated_signals=True,
                                            symmetric_distractor=False) # only need one 
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=config['hparas']['batch_size'], shuffle=False, num_workers=config['num_workers']//2)
    # use anechoic BRIRs for testing
    new_room_manifest = None 
    only14_manifest = None
    
    for idx in range(start,end):
        target_loc = test_dict[idx]['target_loc']
        distract_loc = test_dict[idx]['distract_loc']
        threshold_snr = test_dict[idx]['snr']
        print(test_dict[idx])

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
            all_brir_ixs = only14_manifest['index_brir'].values
            with h5py.File(h5_fn, 'r') as f:
                brir = f['brir'][index_brir]
                if loc == 'target':
                    all_brirs = f['brir'][all_brir_ixs]
            if model_in_sr != sr_src:
                brir = soxr.resample(brir.astype(np.float32), sr_src, model_in_sr)
            ir_dict[loc] = brir.astype(np.float32)

        ########################################################
        # Init level parmas that are constants across all tests  
        # matching human array experiment 
        ########################################################

        # texture transform 
        texture_db = test_dict[idx]['bg_noise_level']
        if texture_db:
            print('Running with texture distractors')
            rms_texture_level = db_to_rms(texture_db)
            set_texture_level = at.BinauralRMSNormalizeForegroundAndBackground(rms_texture_level).cuda()

        # manually set distractors to level, then spatialize, then add 
        dist_level = 65 - 10*np.log10(2) # 2 is the number of distractors
        dist_rms = db_to_rms(dist_level)
        set_dist_level = at.BinauralRMSNormalizeForegroundAndBackground(dist_rms).cuda()

        SNR = threshold_snr
        target_rms = db_to_rms(65 + SNR)
        set_target_level = at.BinauralRMSNormalizeForegroundAndBackground(target_rms).cuda()

        # set cue transform 
        cue_rms = db_to_rms(65)
        set_cue_level = at.BinauralRMSNormalizeForegroundAndBackground(cue_rms).cuda()

        ###################################################
        # Set output file name based on test conditions
        ###################################################

        dist_str = "speech_distractor"
        sym_str = "symmetric"  
        all_stim_str = 'all_stim' if args.run_all_stim else 'subset_stim'
        bg_noise_str = ''
        if args.texture_distractor:
            bg_noise_str = "bg_texture"
        elif args.pink_noise_distractor:
            bg_noise_str = "bg_pink_noise"
            
        texture_str = f"{int(texture_db)}dB_{bg_noise_str}" if texture_db else 'no_texture'

        log_name = f"/{model_name}_cue_{cue_type}_target_loc_{target_loc[0]}_{target_loc[1]}_distract_loc_{distract_loc[0]}_{distract_loc[1]}_{int(threshold_snr)}_SNR_{dist_str}_{texture_str}_{room_str}_{sym_str}_{all_stim_str}"  
        # replace __ with _ in log_name in case format produces it 
        log_name = log_name.replace('__', '_')      
        print(log_name)
        output_name = str(experiment_dir) + log_name + '.pkl'
        print("Overwrite ", args.overwrite)
        # break if file exists and not overwriting
        if not args.overwrite and os.path.exists(output_name):
            continue

        ###################################
        # Setup BRIRs for spatialization
        ###################################
        if texture_db:
            texture_brir  = SpatializeAllLocs(all_brirs[::4], model_sr=model_in_sr).cuda()
        tar_brir = Spatialize(ir_dict['target'], model_sr=model_in_sr).cuda()
        dist_brir_l = Spatialize(ir_dict['distractor_l'], model_sr=model_in_sr).cuda()
        dist_brir_r = Spatialize(ir_dict['distractor_r'], model_sr=model_in_sr).cuda()

        ###################################
        # Arrays for storing results
        ###################################

        output_dict = {'results': None, 'confusions': None}
        accuracies = []
        confusions = []
        pred_list = []
        true_word_int = []
        stim_ix_list = []
        texture_list = []
    
        ###################################
        # Main inference loop
        ###################################

        with torch.no_grad(): 
            for batch in tqdm(dataloader):
                cue, fg, bg, bg_2, label, dist_word_label, dist_word_label2, stim_ixs = batch
                # log ix of stimuli for meta tracking
                stim_ix_list.append(stim_ixs)

                # set levels then spatialize 
                cue, _ = set_cue_level(cue, None)
                target, _ = set_target_level(fg, None)
                bg_1, bg_2 = set_dist_level(bg, bg_2)

                # spatialize signals 
                cue = tar_brir(cue.cuda())
                target = tar_brir(target.cuda())
                bg_1 = dist_brir_l(bg_1.cuda())
                bg_2 = dist_brir_r(bg_2.cuda())

                # combine signals 
                mixture = target + bg_1 + bg_2 

                # get texture stim for this batch
                if texture_db:
                    if args.texture_distractor:
                        diffuse_bg_signal , texture_label = get_texture_eg(texture_dataset)
                        # log the texture used for this batch 
                        texture_list.extend([texture_label] * config['hparas']['batch_size'])
                        # spatialize 
                    elif args.pink_noise_distractor:
                        diffuse_bg_signal = gen_pink_noise_batch(mixture.shape[-1], mixture.shape[0])
                        texture_list.extend(['pink_noise'] * config['hparas']['batch_size'])
                    spatial_texutre = texture_brir(diffuse_bg_signal.cuda())
                    # set texture level 
                    spatial_texture, _ = set_texture_level(spatial_texutre, None)
                    # combine signals 
                    mixture = mixture + spatial_texture
                    # cue = cue + spatial_texture
                else:
                    texture_list.extend([None] * config['hparas']['batch_size'] )  
        
                # get cochleagrams 
                cue, mixture = coch_gram(cue, mixture)

                # pass through model 
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

                # log confusions 
                dist_word_label = dist_word_label.numpy().astype('int')
                dist_word_label2 = dist_word_label2.numpy().astype('int')
                # confusion made if preds == dist_word_label or dist_word_label2
                cons_1 = (preds == dist_word_label).astype('int')
                cons_2 = (preds == dist_word_label2).astype('int')
                cons = np.bitwise_or(cons_1, cons_2) # get union of confusions
                confusions.append(cons)

        # concat test results 
        accuracies = np.concatenate(accuracies)
        preds = np.concatenate(pred_list)
        true_word_int = np.concatenate(true_word_int)
        stim_ix_list = np.concatenate(stim_ix_list)
        output_dict['results'] = accuracies
        output_dict['preds'] = preds
        output_dict['true_word_int'] = true_word_int
        output_dict['stim_ix_list'] = stim_ix_list
        confusions = np.concatenate(confusions)
        output_dict['confusions'] = confusions
        output_dict['textures'] = texture_list

        print(log_name)    
        print(f"Test {distract_loc[0]} azimuth distractor at {threshold_snr} SNR in {room_str}")
        print(f"Accuracy: {accuracies.mean()}")
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
        "--pink_noise_distractor",
        action=argparse.BooleanOptionalAction,
        help="If true, will use noise distractors instead of speech distractors."
    )
    parser.add_argument(
        "--texture_distractor",
        action=argparse.BooleanOptionalAction,
        help="If true, will use textures as distractors instead of speech distractors."
    )
    parser.add_argument(
        "--backbone_with_ecdf_gains",
        action=argparse.BooleanOptionalAction,
        help="Use ecdf gains with backbone architecture",
    )
    args = parser.parse_args()

    run_eval(args)


if __name__ == "__main__":
    cli_main()
