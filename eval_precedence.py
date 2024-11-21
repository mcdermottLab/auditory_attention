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
import torch.nn.functional as F

import argparse
from argparse import ArgumentParser
from corpus.speaker_room_dataset import SpeakerRoomDataset
from tqdm.auto import tqdm
from datetime import datetime

torch.set_float32_matmul_precision('high') # use same as training
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


def precedence_shift_signal(signal, delay_in_frames, lead_side=None, lead_channel=None):
    stack_dim = 0 if signal.ndim == 1 else 1 
    leading_signal = F.pad(signal,(0, delay_in_frames), 'constant', 0)
    trailing_signal = F.pad(signal,(delay_in_frames, 0), 'constant', 0)

    if lead_side == 'left' or lead_channel == 'side':
        stereo_signal = torch.stack([leading_signal, trailing_signal], dim=stack_dim)
    elif lead_side == 'right' or lead_channel == 'center':
        stereo_signal = torch.stack([trailing_signal, leading_signal], dim=stack_dim)
    
    return stereo_signal


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

    # Init flag for diotic test 
    run_diotic = None 

    # load model config 
    config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
    config['num_workers'] = args.n_jobs
    config['hparas']['batch_size'] = 30 # config['data']['loader']['batch_size'] // args.gpus
    # get model input sr for brir resampling
    model_in_sr = config['audio']['rep_kwargs']['sr']

    idx = args.location_idx
    test_dict = pickle.load(open(args.test_manifest, 'rb'))
    n_per_job = args.n_per_job
    start = idx * n_per_job
    end = start + n_per_job

    experiment_dir = f"{args.exp_dir}/{model_name}"
    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)
    model = attn_tracking_lightning.BinauralAttentionModule.load_from_checkpoint(checkpoint_path=checkpoint_path, config=config, strict=False).cuda()
    # define audio transforms to standardize eval transforms across models (v05 skips BinauralCombineWithRandomDBSNR)
    audio_transforms_0_db = at.AudioCompose([
                        at.AudioToTensor(),
                        at.BinauralCombineWithRandomDBSNR(low_snr=0,    # is 0 dB
                                                        high_snr=0), # is 0 dB 
                        at.BinauralRMSNormalizeForegroundAndBackground(rms_level=0.02), # 20 * np.log10(0.02/20e-6) = 60 dB SPL 
                ])

    audio_transforms_0_db = audio_transforms_0_db.cuda()
    # to inference mode 
    model = model.eval()
    coch_gram = model.coch_gram.cuda()

    # set up dataset and dataloader
    dataset = SpeakerRoomDataset(manifest_path='/om2/user/rphess/Auditory-Attention/final_binaural_manifest.pkl',
                                excerpt_path='/om2/user/msaddler/spatial_audio_pipeline/assets/swc/manifest_all_words.pdpkl',
                                cue_type=cue_type,
                                sr=model_in_sr) 
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=config['hparas']['batch_size'], shuffle=False, num_workers=config['num_workers'])

    # use anechoic BRIRs for free-field test
    if args.use_anchoic_46_room:
        new_room_manifest = pd.read_pickle('/om2/user/msaddler/spatial_audio_pipeline/assets/brir/eval/manifest_brir.pdpkl')
        # index_room == 0 is Anechoic room 
        only14_manifest = new_room_manifest[(new_room_manifest['index_room'] == 0)]
        h5_fn = f'/om2/user/msaddler/spatial_audio_pipeline/assets/brir/eval/room0000.hdf5'

    else:
        new_room_manifest = pd.read_pickle('/om2/user/msaddler/spatial_audio_pipeline/assets/brir/ruggles_shinncunningham_2011/manifest_brir.pdpkl')
        only14_manifest = new_room_manifest[(new_room_manifest['index_room'] == 0)]
        h5_fn = f'/om2/user/msaddler/spatial_audio_pipeline/assets/brir/ruggles_shinncunningham_2011/room0000.hdf5'

    for idx in range(start,end):
        target_loc = test_dict[idx]['target_loc']
        distract_loc = test_dict[idx]['distract_loc']
        threshold_snr = test_dict[idx]['snr']
        threshold_snr = threshold_snr if isinstance(threshold_snr, str) else int(threshold_snr)
        run_diotic = test_dict[idx]['diotic']
        lead_channel = test_dict[idx]['lead_channel']
        target_lead_channel = test_dict[idx].get('target_lead_channel', None)
        
        if target_lead_channel:
            log_name = f"/{model_name}_cue_{cue_type}_target_loc_{target_loc[0]}_{target_loc[1]}_distract_loc_{distract_loc[0]}_{distract_loc[1]}_{lead_channel}_lead_{threshold_snr}_SNR_{target_lead_channel}_target_lead"        
        else:
            log_name = f"/{model_name}_cue_{cue_type}_target_loc_{target_loc[0]}_{target_loc[1]}_distract_loc_{distract_loc[0]}_{distract_loc[1]}_{lead_channel}_lead_{threshold_snr}_SNR"        
        
        if run_diotic:
            # in diotic case, distractor location is lead_channel side with leading presentation of diotic distractor
            # either left, center, or right 
            log_name = f"/{model_name}_cue_{cue_type}_diotic_target_loc_{target_loc}_distract_loc_{lead_channel}_{threshold_snr}_SNR"        

        print(log_name)

        audio_transforms_test_db = at.AudioCompose([
                        at.AudioToTensor(),
                        at.BinauralCombineWithRandomDBSNR(low_snr=threshold_snr,    
                                                        high_snr=threshold_snr), 
                        at.BinauralRMSNormalizeForegroundAndBackground(rms_level=0.02), # 20 * np.log10(0.02/20e-6) = 60 dB SPL 
                ])
        audio_transforms_test_db = audio_transforms_test_db.cuda()

        output_name = str(experiment_dir) + log_name + '.pkl'
        if idx % 10 == 0:
            print("Overwrite ", args.overwrite)
        if not args.overwrite and os.path.exists(output_name):
            continue
        
        if not run_diotic:
            ir_dict = dict()
            for loc in ['target', 'distractor']:
                if loc == 'target':
                    coords = target_loc
                elif loc == 'distractor':
                    coords = distract_loc.copy()
                    if coords[0] < 0:
                        coords[0] += 360 

                df_row = only14_manifest[(only14_manifest['src_azim'] == coords[0]) & (only14_manifest['src_elev'] == coords[1])]
                index_brir = df_row['index_brir'].values[0]
                sr_src = df_row['sr'].values[0]
                with h5py.File(h5_fn, 'r') as f:
                    brir = f['brir'][index_brir]
                if model_in_sr != sr_src:
                    brir = soxr.resample(brir.astype(np.float32), sr_src, model_in_sr)
                ir_dict[loc] = brir.astype(np.float32)

            tar_brir = Spatialize(ir_dict['target'], model_sr=model_in_sr).cuda()
            dist_brir = Spatialize(ir_dict['distractor'], model_sr=model_in_sr).cuda()

        ##  precedence effect displacement.  Done by introducing a small delay between left and right sound presentations
        delay_in_s = 0.004 # 4ms in Freyman experiments 
        delay_in_frames = int(delay_in_s * model_in_sr)

        output_dict = {'results': None, 'confusions': None}
        accuracies = []
        confusions = []
        pred_list = []
        true_word_int = []

        with torch.no_grad(): 
            for batch in tqdm(dataloader):
                cue, fg, bg, label, confusion = batch
                if run_diotic:
                    if target_loc == 'center':
                        cue = cue.repeat(2,1,1).permute(1,0,2).cuda()
                        foreground = fg.repeat(2,1,1).permute(1,0,2).cuda()
                        foreground = F.pad(foreground, (0, delay_in_frames))
                    else:
                        cue = precedence_shift_signal(cue,
                                                    delay_in_frames,
                                                    lead_side=target_loc).cuda()

                        foreground = precedence_shift_signal(fg,
                                                    delay_in_frames,
                                                    lead_side=target_loc).cuda()
                    if lead_channel == 'center':
                        background = bg.repeat(2,1,1).permute(1,0,2).cuda()
                        background = F.pad(background, (0, delay_in_frames))

                    else:
                        background = precedence_shift_signal(bg,
                                                    delay_in_frames,
                                                    lead_side=lead_channel).cuda()            
                else:     
                    # spatialize signals 
                    cue = tar_brir(cue.cuda())
                    if not target_lead_channel:
                        foreground = tar_brir(fg.cuda())
                    elif target_lead_channel:            
                        # make precedence cue 
                        cue_w_delay = precedence_shift_signal(cue,
                                    delay_in_frames,
                                    lead_channel=target_lead_channel).cuda()
                        cue_side = dist_brir(cue_w_delay[:, 0, :])
                        cue_center = tar_brir(cue_w_delay[:, 1, :])
                        cue, _ = audio_transforms_0_db(cue_side, cue_center)
                        # Make precedence foreground
                        fg_w_delay = precedence_shift_signal(fg,
                                    delay_in_frames,
                                    lead_channel=target_lead_channel).cuda()
                        fg_side = dist_brir(fg_w_delay[:, 0, :]) # 0 is for side channel
                        fg_center = tar_brir(fg_w_delay[:, 1, :]) # 1 is for center channel 
                        foreground, _ = audio_transforms_0_db(fg_side, fg_center)
                    # Make precedence distractors
                    if lead_channel:
                        bg_w_delay = precedence_shift_signal(bg,
                                                            delay_in_frames,
                                                            lead_channel=lead_channel).cuda()
                        bg_side = dist_brir(bg_w_delay[:, 0, :]) # 0 is for side channel
                        bg_center = tar_brir(bg_w_delay[:, 1, :]) # 1 is for center channel 
                        background, _ = audio_transforms_0_db(bg_side, bg_center)
                    else:
                        background = dist_brir(bg.cuda())
                    
                ## set to desired SNR and SPL 
                cue, _ = audio_transforms_0_db(cue, None)
                # Set foreground to distractor at given SNR, then set to 60dB  
                mixture, _ = audio_transforms_test_db(foreground, background)
                
                cue, mixture = coch_gram(cue, mixture)
                logits = model(cue, mixture, None)
                # Unpack desired metrics 
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

        # Prep output then save for this test 
        output_dict['results'] = accuracies
        output_dict['confusions'] = confusions
        output_dict['preds'] = preds
        output_dict['true_word_int'] = true_word_int
        
        print(f"Test {distract_loc[0]} azimuth distractor at {threshold_snr} SNR")
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
    parser.add_argument(
        "--use_anchoic_46_room",
        action=argparse.BooleanOptionalAction,
        help="If true, will use anechoic BRIRs for free-field test",
    )

    args = parser.parse_args()

    run_eval(args)


if __name__ == "__main__":
    cli_main()
