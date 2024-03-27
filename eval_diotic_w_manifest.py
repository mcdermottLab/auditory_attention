import h5py
import numpy as np
import os
import pandas as pd
import pathlib
import pickle
import scipy.stats as stats
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

def run_eval(args):
    model_name = args.model_name
    checkpoint_path = args.ckpt_path

    ## Load test manifest info
    loc_dict = pickle.load(open(args.test_manifest, 'rb'))
    # unpack 
    test_meta_dict = loc_dict[args.job_idx]
    snr = test_meta_dict.get('snr', 0)
    time_reversed = test_meta_dict.get('time_reversed', False)

    if time_reversed:
        time_reversed_str = 'time_reversed_'
    else:
        time_reversed_str = ''

    config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
    config['num_workers'] = args.n_jobs
    config['hparas']['batch_size'] = 30 # config['data']['loader']['batch_size'] // args.gpus
    config['noise_kwargs']['low_snr'] = snr
    config['noise_kwargs']['high_snr'] = snr

    # get model input sr for brir resampling
    model_in_sr = config['audio']['rep_kwargs']['sr']
    
    # create log and output names based on test params
    log_name = f"{model_name}_{time_reversed_str}{snr}SNR"        
    print(log_name)
    output_dir = pathlib.Path(args.exp_dir) / model_name 
    output_dir.mkdir(parents=True, exist_ok=True)
    output_name = output_dir / f"{log_name}.pkl"

    # define audio transforms 
    if time_reversed:
        diotic_transforms = at.AudioCompose([
                            at.AudioToTensor(),
                            at.TimeReverseBackgroundSignal(time_dim=[-1]),
                            at.CombineWithRandomDBSNR(low_snr=config['noise_kwargs']['low_snr'],
                                                      high_snr=config['noise_kwargs']['high_snr']), 
                            at.RMSNormalizeForegroundAndBackground(rms_level=0.02),  # 0.02 is the default for CV-based models 
                            at.DuplicateChannel(),
                    ])
    else:
        diotic_transforms = at.AudioCompose([
                            at.AudioToTensor(),
                            at.CombineWithRandomDBSNR(low_snr=config['noise_kwargs']['low_snr'],
                                                      high_snr=config['noise_kwargs']['high_snr']), 
                            at.RMSNormalizeForegroundAndBackground(rms_level=0.02),  # 0.02 is the default for CV-based models 
                            at.DuplicateChannel(),
                    ])
                    
    diotic_transforms = diotic_transforms.cuda()

    experiment_dir = f"{args.exp_dir}/{model_name}"
    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)

    model = attn_tracking_lightning.BinauralAttentionModule.load_from_checkpoint(checkpoint_path=checkpoint_path, config=config, strict=False).cuda()
    model = model.eval()
    coch_gram = model.coch_gram.cuda()

    # set up dataset and dataloader
    dataset = SpeakerRoomDataset(manifest_path='/om2/user/rphess/Auditory-Attention/final_binaural_manifest.pkl',
                                excerpt_path='/om2/user/msaddler/spatial_audio_pipeline/assets/swc/manifest_all_words.pdpkl',
                                cue_type='voice_and_location',
                                sr=model_in_sr) 

    def single_signal_collate_fn(batch):
        #apply transforsms to batch
        cues = torch.stack([diotic_transforms(cue, None)[0] for cue, fg, bg, label, confusion in batch])
        mixtures = torch.stack([diotic_transforms(fg, bg)[0] for cue, fg, bg, label, confusion in batch]).type(torch.FloatTensor)
        labels = torch.tensor([label for cue, fg, bg, label, confusion in batch]).type(torch.LongTensor)
        confusion = torch.tensor([confusion for cue, fg, bg, label, confusion in batch]).type(torch.LongTensor)
        return cues, mixtures, labels, confusion

    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=config['hparas']['batch_size'],
                                             shuffle=False,
                                             num_workers=config['num_workers'],
                                             collate_fn=single_signal_collate_fn)


    output_dict = {'results': None, 'confusions': None}
    accuracies = []
    confusions = []
    pred_list = []
    true_word_int = []

    with torch.no_grad(): 
        for batch in tqdm(dataloader):
            cue, mixture, label, confusion = batch
            cue, mixture = coch_gram(cue.cuda(), mixture.cuda())
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
        "--test_manifest",
        default=pathlib.Path(""),
        type=pathlib.Path,
        help="path manifest of target and distractor conditions to use for evaluation",
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
        "--job_idx",
        type=int,
        help="index into test manifest dictionary",
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
