# Following is originally copied from PyTorch RNN-T ASR Example:
# https://github.com/pytorch/audio/tree/820b383b3b21fc06e91631a5b1e6ea1557836216/examples/asr/librispeech_emformer_rnnt

import pathlib
from argparse import ArgumentParser
import yaml
import pickle
import torch
import re

from pytorch_lightning import Trainer, seed_everything
# from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
# from src.attn_tracking_lightning import AttentionalTrackingModule
# from src.attentional_tracking_control_lightning import AttnTrackingControlModule
# from src.attn_rove_rms_lightning import AttnRoveRMSModule

seed_everything(1)

task_name_dict = {
    'all_targets_harmonic_single_distractor_0dB_SNR_jitter_fn_render.pdpkl': "_harmonic_speech_jitter_render_",
    'all_targets_whispered_single_distractor_0dB_SNR.pdpkl': "_whispered_speech_",
    'all_targets_inharmonic_single_distractor_0dB_SNR.pdpkl': "_inharmonic_speech_",
    'harmonic_target_inharmonic_distractor_0dB_SNR.pdpkl': "_harmonic_target_inharmonic_distractor_",
    'inharmonic_target_harmonic_distractor_0dB_SNR.pdpkl': "_inharmonic_target_harmonic_distractor_",
    'harmonic_target_whispered_distractor_0dB_SNR.pdpkl': "_harmonic_target_whispered_distractor_",
    'whispered_target_harmonic_distractor_0dB_SNR.pdpkl': "_whispered_target_harmonic_distractor_",
    'inharmonic_target_whispered_distractor_0dB_SNR.pdpkl': "_inharmonic_target_whispered_distractor_",
    'whispered_target_inharmonic_distractor_0dB_SNR.pdpkl': "_whispered_target_inharmonic_distractor_",
}

def run_eval(args):
    model_name = args.model_name
    checkpoint_path = args.ckpt_path
    config = yaml.load(open(args.config_name, 'r'), Loader=yaml.FullLoader)
    # read test manifest
    if args.test_manifest.as_posix() != "":
        # open pickle 
        with open(args.test_manifest, 'rb') as f:
            eval_conditions = pickle.load(f)
            cond_dict = eval_conditions[args.array_id]
        harmonic = cond_dict.get('harmonic', False)
        whispered = cond_dict.get('whispered', False)
        inharmonic = cond_dict.get('inharmonic', False)
        clean_targets = cond_dict.get('clean_targets', False)
        manifest = cond_dict.get('manifest', None)
    else:
        harmonic = args.harmonic
        whispered = args.whispered
        inharmonic = args.inharmonic
        clean_targets = args.clean_targets

    config['matched_cue_level'] = False
    if 'cv' in model_name:
        config['corpus']['root'] = cv_eval_h5_path
        config['loader']['num_workers'] = args.n_jobs
        config['loader']['batch_size'] = 32
    elif 'binaural' in model_name.lower() or 'mono' in model_name.lower() or 'v0' in model_name.lower():
        config['hparas']['batch_size'] = 1
        config['num_workers'] = args.n_jobs
        config['data'] = {}
        config['data']['corpus'] = {}
        config['audio']['rep_kwargs']['center_crop'] = True
    else:
        config['data']['loader']['num_workers'] = args.n_jobs
        config['data']['loader']['batch_size'] = 1 # config['data']['loader']['batch_size'] // args.gpus
    config['corpora_name'] = 'TIMIT'
    config['data']['corpus']['clean_targets'] = clean_targets
    snr = 'clean' if clean_targets else '0dB_SNR'

    if args.test_manifest.as_posix() != "" and manifest is not None:
        config['data']['corpus']['root'] = manifest
        task_name = task_name_dict[manifest.split('/')[-1]]

    elif args.test_manifest.as_posix() != "" and manifest is None:
        if harmonic: 
            config['data']['corpus']['root'] = '/om2/user/imgriff/datasets/timit/harmonic_timit/all_targets_harmonic_single_distractor_0dB_SNR_jitter_fn_render.pdpkl'
            task_name = "_harmonic_speech_jitter_render_"    
            
        elif whispered:
            config['data']['corpus']['root'] = '/om2/user/imgriff/datasets/timit/whispered_timit/all_targets_whispered_single_distractor_0dB_SNR.pdpkl'
            task_name = "_whispered_speech_"
            
        elif inharmonic:
            config['data']['corpus']['root'] = '/om2/user/imgriff/datasets/timit/inharmonic_timit/all_targets_inharmonic_single_distractor_0dB_SNR.pdpkl'
            task_name = "_inharmonic_speech_"
            
    else:
        if clean_targets:
            config['data']['corpus']['root'] = '/om2/user/imgriff/datasets/timit/clean_timit_targets_attn_task_0.1rms.pdpkl'
        else:
            config['data']['corpus']['root'] = '/om2/user/imgriff/datasets/timit/attn_task_dataframes/timit_attn_stim_for_model_all_targets.pdpkl'
            snr = ''
        task_name = "_"

    if "binaural" in model_name.lower() or "mono" in model_name.lower():
        config['corpus']['root'] = config['data']['corpus']['root']
        config['corpus']['clean_targets'] = config['data']['corpus']['clean_targets']
        if "mono" in model_name.lower():
            config['corpus']['run_mono'] = True

    log_name = f"TIMIT{task_name}attn_task_{snr}_all_targets_{model_name}"


    print(log_name)

#     checkpoint_dir = args.exp_dir / "checkpoints"
#     print(checkpoint_dir)
#     # get latest checkpoint
#     checkpoints = sorted(checkpoint_dir.glob('*.ckpt'))
#     checkpoint_path = checkpoints[-1] # sort by training step & get latest

    # get eval set & update params 
#     test_condition = eval_conditions[args.array_id]
    experiment_dir = args.exp_dir
    
    #config['data']['corpus']['root'] = f'/om2/user/mjmcp/TestSets/{test_condition}'
    #model_name = config['model_name']

    logger = CSVLogger(experiment_dir, name=log_name)

    if torch.__version__ >= '2.0.0':
        trainer = Trainer(
            default_root_dir=experiment_dir,
            deterministic=True,
            num_nodes=args.num_nodes,
            devices=args.gpus,
            accelerator="gpu" if args.gpus > 0 else 'cpu',
            logger=logger
        )
    else:
        trainer = Trainer(
            default_root_dir=experiment_dir,
            deterministic=True,
            num_nodes=args.num_nodes,
            gpus=args.gpus,
            accelerator="gpu" if args.gpus > 0 else 'cpu',
            logger=logger
        )  
    
    # load model checkpoint 
    print(checkpoint_path)
    from src.spatial_attn_lightning import BinauralAttentionModule
    model = BinauralAttentionModule.load_from_checkpoint(checkpoint_path=checkpoint_path, config=config)
    # evaluate model  
    trainer.test(model)


def cli_main():
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default="", help='Path to experiment config.')
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
        default='MultiDistractorAttnCNN',
        type=str,
        help="Name of model to use in file name.",
    )
    parser.add_argument(
        "--config_name",
        default=pathlib.Path("config/attentional_cue/attn_cue_lr_1e-4_bs_64_constrained_slope_multi_distractor.yaml"),
        type=pathlib.Path,
        help="Config file used to specify model parameters",
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
        "--array_id",
        default=0,
        type=int,
        help="Slurm array task ID",
    )
    parser.add_argument(
        "--test_manifest",
        default=pathlib.Path(""),
        type=pathlib.Path,
        help="Path to config for test manifest mapping to array_id",
    )
    parser.add_argument(
        "--harmonic",
        default=False,
        action='store_true',
        help="run using harmonic speech",
    )  
    parser.add_argument(
        "--whispered",
        default=False,
        action='store_true',
        help="run using whispered speech",
    )    
    parser.add_argument(
        "--inharmonic",
        default=False,
        action='store_true',
        help="run using inharmonic speech",
    )    
    parser.add_argument(
        "--clean_targets",
        default=False,
        action='store_true',
        help="run without distractors",
    )   
    
    args = parser.parse_args()

    run_eval(args)


if __name__ == "__main__":
    cli_main()