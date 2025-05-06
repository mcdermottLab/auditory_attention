import pathlib
from argparse import ArgumentParser, BooleanOptionalAction
import yaml
import pickle
import csv
import torch 
import os, sys
import json
import numpy as np 
import pandas as pd
from tqdm.auto import tqdm
from pytorch_lightning import seed_everything
from src.attn_tracking_lightning import AttentionalTrackingModule
from src.spatial_attn_lightning import BinauralAttentionModule 
from corpus.swc_mono_test import SWCMonoTestSet, SWCMonoTestSet2024, SWCMonoTestSetH5Dataset
import src.audio_transforms as at

sys.path.append('phaselocknet_torch')
from phaselocknet_torch import phaselocknet_model
from phaselocknet_torch import util

seed_everything(1)

torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
    
def run_eval(args):

    model_name = "Saddler_Arch"

    dir_model = "/om2/user/msaddler/auditoryutil/models/singletask/batchnorm_arch0_0000_taskW/"

    # Option 1: Manually construct the model from config files:
    with open(os.path.join(dir_model, "config.json"), "r") as f:
        config_model = json.load(f)
    with open(os.path.join(dir_model, "arch.json"), "r") as f:
        architecture = json.load(f)
    model = phaselocknet_model.Model(
        config_model=config_model,
        architecture=architecture,
        input_shape=[2, 110_250, 2],  # <-- [batch, timesteps @ 50 kHz sampling rate, channels==2] for sound_localization
        config_random_slice={"size": [50, 20000], "buffer": [0, 500]},
    )

    # Load model weights from torch checkpoint
    util.load_model_checkpoint(
        model=model.perceptual_model,
        dir_model=dir_model,
        fn_ckpt="ckpt_BEST.pt",
        weights_only=True,
    )

    model = model.cuda().eval()

    # set audio transforms
    model_sr = config_model['kwargs_cochlea']['sr_input']

    # get snr for audio transforms if part of the config
    with open(args.stim_cond_map, 'rb') as f:
        condition_dict = pickle.load(f)
    condition, snr = condition_dict[args.array_id]

    audio_transforms = at.AudioCompose([
                at.AudioToTensor(),
                at.Resample(orig_freq=44_100, new_freq=model_sr),
                at.CombineWithRandomDBSNR(low_snr=snr, high_snr=snr), 
                at.RMSNormalizeForegroundAndBackground(rms_level=0.02),  # 0.02 is the default for CV-based models 
                at.DuplicateChannel()
                ])
    label_type = "CV"
    dataset = SWCMonoTestSetH5Dataset(h5_path=args.stim_path,
                                    eval_distractor_cond=condition,
                                    model_sr=model_sr,
                                    label_type=label_type)

    print(f"Evaluating {model_name} on {condition} at {snr}db SNR")

    def collate_fn(batch):
        mixtures = []
        labels = []
        for _, target, distractor, tgt_label, dist_label in batch:
            mixture, _ = audio_transforms(target, distractor)
            mixture = mixture.T.reshape(1,-1, 2)
            mixtures.append(mixture)
            labels.append(tgt_label)
        mixtures = torch.cat(mixtures, dim=0)
        labels = torch.tensor(labels)
        return None, mixtures, labels

    print(dataset)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=8,
                                             shuffle=False,
                                             collate_fn=collate_fn,
                                             num_workers=args.n_jobs)


    out_dir = args.exp_dir / model_name 
    out_name = out_dir / f"{model_name}_{condition}_{snr}dB_SNR_eval_results.csv" 
    print(f"Output directory: {out_dir}")
    # make dir if it doesn't exist
    out_dir.mkdir(parents=True, exist_ok=True)
    # track running average of accuracy and confusions 
    acc_sum = 0

    if out_name.exists() and not args.overwrite:
        # if any([arch_ix in model_name for arch_ix in ['9', '12', '6', '8']]):
        #     pass 
        # else:
        print(f"File {out_name} already exists. Exiting.")
        return 
    
    with open(out_name, 'w') as file:
        csv_out = csv.writer(file, delimiter=",")
        # write column header to csv
        print("writing csv header")
        if ('2024' in str(args.stim_path) or args.spotlight_expmnt) and not args.full_h5_stim_set and not 'popham' in str(args.stim_path):        
            csv_out.writerow(['pred_word_int', 'true_word_int', 'accuracy', 'stim_name'])
        else:
            csv_out.writerow(['pred_word_int', 'true_word_int', 'accuracy'])

        # run eval 
        for i, batch in enumerate(tqdm(dataloader, desc=f"evaluating {model_name} on {condition} at {snr}dB SNR")):
            if '2024' in str(args.stim_path) and not args.full_h5_stim_set and not 'popham' in str(args.stim_path):
                cue, mixture, word, stim_tag = batch
            else:
                cue, mixture, word = batch

            # to device 
            mixture = mixture.cuda()

            logits = model(mixture)['label_word_int']
            
            # -1 from preds to map back from Mark's labels to ours
            preds = logits.softmax(dim=-1).argmax(dim=-1).cpu().detach().numpy().astype('int') - 1 
            true_word = word.numpy().astype('int')
            accuracy = (true_word == preds).astype('int')
            acc_sum += accuracy.sum()
            # write to csv
            if ('2024' in str(args.stim_path) or args.spotlight_expmnt) and not args.full_h5_stim_set and not 'popham' in str(args.stim_path):        
                rows = list(zip(*[preds, true_word, accuracy, stim_tag]))
            else:
                rows = list(zip(*[preds, true_word, accuracy]))
            csv_out.writerows(rows)
            if i == 0:
                print(f"EG of data writing: {rows}")
            if i % 100 == 0:
                print(f"writing on batch {i} of {len(dataloader)}")
                file.flush() # only write every 100 batches 
        # print final accuracy
        acc = acc_sum / len(dataset)
        print(f"Final accuracy: {acc}")
        
def cli_main():
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default="", help='Path to experiment config.')
    parser.add_argument('--config_list_path', type=str, default="", help='Path to experiment config pkl file.')
    parser.add_argument(
        "--exp_dir",
        default=pathlib.Path("./exp"),
        type=pathlib.Path,
        help="Directory to save test results in. (Default: './exp')",
    )
    parser.add_argument(
        "--stim_path",
        default=pathlib.Path("/om/user/imgriff/datasets/human_word_rec_SWC_2023/sounds/"),
        type=pathlib.Path,
        help="Path where background stimuli are saved",
    )
    parser.add_argument(
        "--stim_cond_map",
        default=None,
        help="Path to pickle file containing condition map for stimuli",
    )
    parser.add_argument(
        "--ckpt_path",
        default=pathlib.Path("./exp"),
        type=pathlib.Path,
        help="path to checkpoint (Default: './exp')",
    )
    parser.add_argument(
        "--n_jobs",
        default=0,
        type=int,
        help="Number of CPUs for dataloader. (Default: 0)",
    )
    parser.add_argument(
        "--array_id",
        default=0,
        type=int,
        help="Slurm array task ID",
    )  
    parser.add_argument(
        "--full_h5_stim_set",
        action='store_true',
        help="If set, load full h5 stimulus set",
    )
    parser.add_argument(
        "--spotlight_expmnt",
        action='store_true',
        help="If set, load spotlight experiment stimuli",
    )
    parser.add_argument(
        "--overwrite",
        action=BooleanOptionalAction,
        help="If true, will overwrite existing results",
    )
    parser.add_argument(
        "--backbone_with_ecdf_gains",
        action=BooleanOptionalAction,
        help="Use ecdf gains with backbone architecture",
    )
    parser.add_argument(
        "--backbone_with_ecdf_feature_gains",
        action=BooleanOptionalAction,
        help="Use ecdf gains with backbone architecture",
    )
    args = parser.parse_args()

    run_eval(args)


if __name__ == "__main__":
    cli_main()
