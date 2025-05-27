import pathlib
from argparse import ArgumentParser, BooleanOptionalAction
import yaml
import pickle
import csv
import torch 
import os 
import numpy as np 
import pandas as pd
from tqdm.auto import tqdm
from pytorch_lightning import seed_everything
from src.spatial_attn_lightning import BinauralAttentionModule 

from corpus.swc_mono_test import SWCMonoTestSet, SWCMonoTestSet2024, SWCMonoTestSetH5Dataset
import src.audio_transforms as at

seed_everything(1)

torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
    
def run_eval(args):
    if args.config != "":
        config_path = pathlib.Path(args.config)
        checkpoint_path = args.ckpt_path
        test_idx = args.array_id

    elif args.config_list_path != "":
        with open(args.config_list_path, 'rb') as f:
            config_dict = pickle.load(f)
            config_output = config_dict[args.array_id]
            if isinstance(config_output, dict):
                print(f"Loading config from {config_output['config_path']}")
                config_path, checkpoint_path, test_idx = config_output['config_path'], config_output['ckpt_path'], config_output['test_idx']
                config_path = pathlib.Path(config_path)
            else:
                config_path = pathlib.Path(config_output)
                checkpoint_path = args.ckpt_path
                test_idx = args.array_id

    config = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)
    config_str_name = str(config_path)
    print(config_str_name)
    model_name = config_path.stem

    strict_ckpt = True
    backbone_str_modifier = ''
    if 'backbone' in model_name and 'learned' not in model_name:
        if args.backbone_with_ecdf_gains:
            config['model']['backbone_with_ecdf_gains'] = True
            backbone_str_modifier = '_ecdf_gains'
            strict_ckpt = False

        elif args.backbone_with_ecdf_feature_gains:
            config['model']['backbone_with_ecdf_gains'] = True
            config['model']['gain_type'] = 'ECDFFeatureGains'
            backbone_str_modifier = '_ecdf_feature_gains'
            strict_ckpt = False
        else:
            config['model']['backbone_with_ecdf_gains'] = False
            backbone_str_modifier = '_no_gains'


    # handle checkpoint path - if not provided, get latest 
    if checkpoint_path == "":
        ckpt_dir = pathlib.Path('attn_cue_models/') / model_name / 'checkpoints'
        checkpoint_path = sorted(ckpt_dir.glob("*.ckpt"), key=os.path.getctime)[-1]

    print(f"Loading model from {checkpoint_path}")
    
    # load model 
    if 'saddler' in config_str_name:
        module = SaddlerBackBoneModule
        label_type = 'CV'
    elif 'binaural_attn' in config_str_name or 'word_task' in config_str_name:
        module = BinauralAttentionModule
        label_type = 'CV'

    else:
        module = AttentionalTrackingModule
        config['data']['audio']['rep_kwargs']['center_crop'] = True
        config['data']['audio']['rep_kwargs']['out_dur'] = 2
        label_type = "WSN"
    
    dual_task_arch =  config['model'].get("cue_loc_task", False)

    # set audio transforms
    sr = config['audio']['rep_kwargs']['sr']
    audio_config = config['audio']

    # get snr for audio transforms if part of the config
    if args.full_h5_stim_set:
        with open(args.stim_cond_map, 'rb') as f:
            condition_dict = pickle.load(f)
        condition, snr = condition_dict[test_idx]

    IIR_COCH = not audio_config['rep_kwargs']['rep_on_gpu']

    if IIR_COCH:
        audio_transforms = at.AudioCompose([
                at.AudioToTensor(),
                # at.CombineWithRandomDBSNR(low_snr=snr, high_snr=snr), 
                at.RMSNormalizeForegroundAndBackground(rms_level=0.1),  # 0.1 is the default for the swc-based models 
                at.UnsqueezeAudio(dim=0),
                at.AudioToAudioRepresentation(**audio_config),
            ])
    if 'mono' not in config_str_name:
        print(f"Using diotic input")
        if args.full_h5_stim_set:
            audio_transforms = at.AudioCompose([
                        at.AudioToTensor(),
                        at.CombineWithRandomDBSNR(low_snr=snr, high_snr=snr), 
                        at.RMSNormalizeForegroundAndBackground(rms_level=0.02),  # 0.02 is the default for CV-based models 
                        at.UnsqueezeAudio(dim=0),
                        at.DuplicateChannel()
                        ])
        else:
            audio_transforms = at.AudioCompose([
                        at.AudioToTensor(),
                        at.RMSNormalizeForegroundAndBackground(rms_level=0.02),  # 0.02 is the default for CV-based models 
                        at.UnsqueezeAudio(dim=0),
                        at.DuplicateChannel()
                ])
        if args.spotlight_expmnt:
            audio_transforms = at.AudioCompose([
                        at.AudioToTensor(),
                        at.BinauralRMSNormalizeForegroundAndBackground(rms_level=0.02,
                                                                       v2_demean=True), # 20 * np.log10(0.02/20e-6) = 60 dB SPL 
                ])
    else:
        audio_transforms = at.AudioCompose([
                    at.AudioToTensor(),
                    at.RMSNormalizeForegroundAndBackground(rms_level=0.02),  # 0.02 is the default for CV-based models 
                    at.UnsqueezeAudio(dim=0),
            ])  
    
    if module == SaddlerBackBoneModule:
        audio_transforms = at.AudioCompose([
                            at.AudioToTensor(),
                            at.Resample(orig_freq=44_100, new_freq=50_000),
                            at.CombineWithRandomDBSNR(low_snr=snr, high_snr=snr), 
                            at.RMSNormalizeForegroundAndBackground(rms_level=0.02),  # 0.02 is the default for CV-based models 
                            at.UnsqueezeAudio(dim=0),
                            at.DuplicateChannel()# 20 * np.log10(0.02/20e-6) = 60 dB SPL 
        ])

    # load and freeze model
    model = module.load_from_checkpoint(checkpoint_path=checkpoint_path,
                                        config=config,
                                        strict=strict_ckpt).eval().cuda()
    use_coch = True if ('v0' in config_str_name or 'word_task' in config_str_name) and 'saddler' not in config_str_name else False 
    coch_gram = None
    if use_coch:
        coch_gram = model.coch_gram.cuda()

    if args.stim_cond_map and not args.spotlight_expmnt or args.full_h5_stim_set:
        if args.full_h5_stim_set:
            dataset = SWCMonoTestSetH5Dataset(h5_path=args.stim_path,
                                            eval_distractor_cond=condition,
                                            model_sr=sr,
                                            label_type=label_type)
        else:
            dataset = SWCMonoTestSet(stim_path=args.stim_path,
                            cond_ix=test_idx,
                            model_sr=sr,
                            label_type=label_type,
                            stim_cond_map=args.stim_cond_map)

            condition, snr = dataset.stim_cond_map[test_idx]

    elif '2024' in str(args.stim_path) or args.spotlight_expmnt:
        dataset = SWCMonoTestSet2024(stim_path=args.stim_path,
                                cond_ix=test_idx,
                                model_sr=sr,
                                label_type=label_type,
                                stim_cond_map=args.stim_cond_map)
        if args.spotlight_expmnt:
            condition_dict = dataset.stim_cond_map[test_idx]
            condition = f"target_azim_{condition_dict['target_azim']}_distractor_azim_{condition_dict['distractor_azim']}"
            snr = 0
        else:
            condition, snr = dataset.stim_cond_map[test_idx]

    elif 'popham' in str(args.stim_path):
        dataset = SWCMonoTestSet(stim_path=args.stim_path,
                                cond_ix=test_idx,
                                model_sr=sr,
                                label_type=label_type,
                                popham_stim=True)  
        condition_dict = dataset.stim_cond_map[test_idx]
        target_harm = condition_dict['target_harmonicity']
        dist_harm = condition_dict['distractor_harmonicity']
        dist_harm = "no" if dist_harm is None else dist_harm
        condition = f"{target_harm}_target_{dist_harm}_distractor"
        snr = 0 

    else:
        dataset = SWCMonoTestSet(stim_path=args.stim_path,
                                cond_ix=test_idx,
                                model_sr=sr,
                                label_type=label_type,
                                unfamiliar_distractor = True if 'language' in args.stim_path.as_posix() else False)
        
        condition, snr = dataset.stim_cond_map[test_idx]
    print(f"Evaluating {model_name} on {condition} at {snr}db SNR")
    if args.full_h5_stim_set:
        def collate_fn(batch):
            #apply transforms to batch
            cues = torch.stack([audio_transforms(cue, None)[0] for cue, _, _, _, _ in batch])
            mixtures = torch.stack([audio_transforms(target, bg)[0] for _, target, bg, _, _ in batch])
            labels = torch.tensor([label for _, _, _, label, _ in batch]).type(torch.LongTensor)
            return cues, mixtures, labels

    elif ("2024" in str(args.stim_path) or args.spotlight_expmnt) and not 'popham' in str(args.stim_path):
        def collate_fn(batch):
            #apply transforms to batch
            cues = torch.stack([audio_transforms(cue, None)[0] for cue, _, _, _ in batch])
            mixtures = torch.stack([audio_transforms(mix, None)[0] for _, mix,  _, _ in batch])
            labels = torch.tensor([label for _, _, label, _ in batch]).type(torch.LongTensor)
            stim_tag = [stim_tag for _, _, _, stim_tag in batch]
            return cues, mixtures, labels, stim_tag
    else:
        def collate_fn(batch):
            #apply transforms to batch
            cues = torch.stack([audio_transforms(cue, None)[0] for cue, _, _ in batch])
            mixtures = torch.stack([audio_transforms(mix, None)[0] for _, mix,  _ in batch])
            labels = torch.tensor([label for _, _, label in batch]).type(torch.LongTensor)
            return cues, mixtures, labels
    if module == SaddlerBackBoneModule:
        def collate_fn(batch):
            cues = []
            mixtures = []
            labels = []
            for cue, target, distractor, tgt_label, dist_label in batch:
                cue, _ = audio_transforms(cue, None)
                mixture, _ = audio_transforms(target, distractor)
                mixture = mixture.T.reshape(1,-1, 2)
                cue = cue.T.reshape(1,-1, 2)
                cues.append(cue)
                mixtures.append(mixture)
                labels.append(tgt_label)
            cues = torch.cat(cues, dim=0)
            mixtures = torch.cat(mixtures, dim=0)
            labels = torch.tensor(labels)
            return cues, mixtures, labels

    print(dataset)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=8,
                                             shuffle=False,
                                             collate_fn=collate_fn,
                                             num_workers=args.n_jobs)

    # set up output file 
    if 'backbone' in model_name:
        out_dir = args.exp_dir / f"{model_name}{backbone_str_modifier}"
        out_name = out_dir / f"{model_name}{backbone_str_modifier}_{condition}_{snr}dB_SNR_eval_results.csv" 
    else:
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
            cue = cue.cuda()
            mixture = mixture.cuda()

            if coch_gram: # if cochleagram is not part of model arch. 
                cue, mixture = coch_gram(cue, mixture)

            if module == BinauralAttentionModule or module == SaddlerBackBoneModule:
                logits = model(cue, mixture, None)
            else:
                logits = model(cue, mixture)
                
            if dual_task_arch:
                logits, _ = logits # unpack word and location tuple 

            preds = logits.softmax(dim=-1).argmax(dim=-1).cpu().detach().numpy().astype('int')
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
