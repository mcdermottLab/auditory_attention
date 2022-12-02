# Following is originally copied from PyTorch RNN-T ASR Example:
# https://github.com/pytorch/audio/tree/820b383b3b21fc06e91631a5b1e6ea1557836216/examples/asr/librispeech_emformer_rnnt

import pathlib
from argparse import ArgumentParser
import yaml
import pickle

from pytorch_lightning import Trainer, seed_everything
# from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from src.attn_tracking_lightning import AttentionalTrackingModule
# from src.attentional_tracking_control_lightning import AttnTrackingControlModule
# from src.attn_rove_rms_lightning import AttnRoveRMSModule


seed_everything(1)

def run_eval(args):
    
    
    # with open(args.eval_cond_file, 'rb') as f:
    #     eval_conditions = pickle.load(f)
        
    # model_name, snr, num_bg_talkers = eval_conditions[args.array_id]
    
#     model_name = "MultiDistractorAttnCNN"
# # if "AttnCNN" in model_name:
#     config_name = "config/attentional_cue/attn_cue_high_snr_lr_1e-4_bs_64.yaml"
#     if model_name == "AttnCNN":
#         checkpoint_path = "/om2/user/jcruse/projects/End-to-end-ASR-Pytorch/attn_cue_models/attn_cue_jsin_pilot_no_pretrain_bs_64_lr_1e-4/checkpoints/epoch=1-step=120790.ckpt"

#     elif model_name == "AttnCNNConstrained":
#         checkpoint_path = "/om2/user/jcruse/projects/End-to-end-ASR-Pytorch/attn_cue_models/attn_cue_jsin_pilot_no_pretrain_norm_at_input_pos_slope_bs_64_lr_1e-4/checkpoints/epoch=0-step=65000-v1.ckpt"

#     elif model_name == "AttnCNNPosSlope":
#         checkpoint_path = "/om2/user/jcruse/projects/End-to-end-ASR-Pytorch/attn_cue_models/attn_cue_jsin_pilot_no_pretrain_pos_slope_bs_64_lr_1e-4/checkpoints/epoch=1-step=95791.ckpt"

#     elif model_name == "AttnCNNOnlyNorm":
#         checkpoint_path = "/om2/user/jcruse/projects/End-to-end-ASR-Pytorch/attn_cue_models/attn_cue_jsin_pilot_no_pretrain_norm_at_input_bs_64_lr_1e-4/checkpoints/epoch=1-step=135791.ckpt"
            
#     elif model_name == "AttnTrackingControl":
#         config_name = "config/attentional_cue/attn_tracking_control_high_snr.yaml"
#         checkpoint_path = "/om2/user/jcruse/projects/End-to-end-ASR-Pytorch/multi_talker_control/jsin_precombined_gammatone_40_channels_20kHz_on_gpu_1e-4lr/checkpoints/epoch=5-step=741324.ckpt"
        
#     elif model_name == "AudiosetBackground":
#         config_name = "config/attentional_cue/attn_cue_lr_1e-4_bs_64_constrained_slope_noise_only.yaml"
#         checkpoint_path = "/om2/user/imgriff/projects/End-to-end-ASR-Pytorch/attn_cue_models/attn_cue_jsin_audset_bg_fully_constrained_bs_64_lr_1e-4/checkpoints/epoch=1-step=140791.ckpt"
        
#     elif model_name == "MultiDistractorAttnCNN":
#         config_name = "config/attentional_cue/attn_cue_lr_1e-4_bs_64_constrained_slope_multi_distractor.yaml"
#         checkpoint_path = "/om2/user/imgriff/projects/End-to-end-ASR-Pytorch/attn_cue_models/attn_cue_jsin_multi_distractor_w_audioset_bs_64_lr_1e-4/checkpoints/epoch=0-step=70000.ckpt"
    
    model_name = args.model_name
    checkpoint_path = args.ckpt_path
    config = yaml.load(open(args.config_name, 'r'), Loader=yaml.FullLoader)
    
    config['matched_cue_level'] = False
    config['data']['loader']['num_workers'] = args.n_jobs
    config['data']['loader']['batch_size'] = 1 # config['data']['loader']['batch_size'] // args.gpus
    config['corpora_name'] = 'TIMIT'
    config['data']['corpus']['clean_targets'] = args.clean_targets
    snr = 'clean' if args.clean_targets else '0dB_SNR'

    if args.harmonic: 
        config['data']['corpus']['root'] = '/om2/user/imgriff/datasets/timit/harmonic_timit/all_targets_harmonic_single_distractor_0dB_SNR_jitter_fn_render.pdpkl'
        task_name = "_harmonic_speech_jitter_render_"    
        
    elif args.whispered:
        config['data']['corpus']['root'] = '/om2/user/imgriff/datasets/timit/whispered_timit/all_targets_whispered_single_distractor_0dB_SNR.pdpkl'
        task_name = "_whispered_speech_"
        
    elif args.inharmonic:
        config['data']['corpus']['root'] = '/om2/user/imgriff/datasets/timit/inharmonic_timit/all_targets_inharmonic_single_distractor_0dB_SNR.pdpkl'
        task_name = "_inharmonic_speech_"
            
    else:
        if args.clean_targets:
            config['data']['corpus']['root'] = '/om2/user/imgriff/datasets/timit/clean_timit_targets_attn_task_0.1rms.pdpkl'
        else:
            config['data']['corpus']['root'] = '/om2/user/imgriff/datasets/timit/attn_task_dataframes/timit_attn_stim_for_model_all_targets.pdpkl'
            snr = ''
        task_name = "_"
    
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
    if model_name == 'AttnTrackingControl':
        model = AttnTrackingControlModule.load_from_checkpoint(checkpoint_path=checkpoint_path, config=config)    
    elif model_name == "AttnRoveRMSCNN":
        model = AttnRoveRMSModule.load_from_checkpoint(checkpoint_path=checkpoint_path, config=config)
    else:
        model = AttentionalTrackingModule.load_from_checkpoint(checkpoint_path=checkpoint_path, config=config)  
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
        "--get_confusions",
        default=False,
        action='store_true',
        help="get target-distractor confusions",
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
