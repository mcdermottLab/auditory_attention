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
    
    
    with open(args.eval_cond_file, 'rb') as f:
        eval_conditions = pickle.load(f)
        
    model_name, snr, num_bg_talkers = eval_conditions[args.array_id]
    
    
# if "AttnCNN" in model_name:
    config_name = "config/attentional_cue/attn_cue_high_snr_lr_1e-4_bs_64.yaml"
    if model_name == "AttnCNN":
        checkpoint_path = "/om2/user/jcruse/projects/End-to-end-ASR-Pytorch/attn_cue_models/attn_cue_jsin_pilot_no_pretrain_bs_64_lr_1e-4/checkpoints/epoch=1-step=120790.ckpt"

    elif model_name == "AttnCNNConstrained":
        checkpoint_path = "/om2/user/jcruse/projects/End-to-end-ASR-Pytorch/attn_cue_models/attn_cue_jsin_pilot_no_pretrain_norm_at_input_pos_slope_bs_64_lr_1e-4/checkpoints/epoch=0-step=65000-v1.ckpt"

    elif model_name == "AttnCNNPosSlope":
        checkpoint_path = "/om2/user/jcruse/projects/End-to-end-ASR-Pytorch/attn_cue_models/attn_cue_jsin_pilot_no_pretrain_pos_slope_bs_64_lr_1e-4/checkpoints/epoch=1-step=95791.ckpt"

    elif model_name == "AttnCNNOnlyNorm":
        checkpoint_path = "/om2/user/jcruse/projects/End-to-end-ASR-Pytorch/attn_cue_models/attn_cue_jsin_pilot_no_pretrain_norm_at_input_bs_64_lr_1e-4/checkpoints/epoch=1-step=135791.ckpt"
            
    elif model_name == "AttnTrackingControl":
        config_name = "config/attentional_cue/attn_tracking_control_high_snr.yaml"
        checkpoint_path = "/om2/user/jcruse/projects/End-to-end-ASR-Pytorch/multi_talker_control/jsin_precombined_gammatone_40_channels_20kHz_on_gpu_1e-4lr/checkpoints/epoch=5-step=741324.ckpt"
        
    elif model_name == "AudiosetBackground":
        config_name = "config/attentional_cue/attn_cue_lr_1e-4_bs_64_constrained_slope_noise_only.yaml"
        checkpoint_path = "/om2/user/imgriff/projects/End-to-end-ASR-Pytorch/attn_cue_models/attn_cue_jsin_audset_bg_fully_constrained_bs_64_lr_1e-4/checkpoints/epoch=1-step=140791.ckpt"
        
    elif model_name == "MultiDistractorAttnCNN":
        config_name = "config/attentional_cue/attn_cue_lr_1e-4_bs_64_constrained_slope_multi_distractor.yaml"
        checkpoint_path = "/om2/user/imgriff/projects/End-to-end-ASR-Pytorch/attn_cue_models/attn_cue_jsin_multi_distractor_w_audioset_bs_64_lr_1e-4/checkpoints/epoch=0-step=70000.ckpt"
        
    config = yaml.load(open(config_name, 'r'), Loader=yaml.FullLoader)
    
    
    config['data']['loader']['num_workers'] = args.n_jobs
    config['data']['loader']['batch_size'] = 8 # config['data']['loader']['batch_size'] // args.gpus
    config['data']['corpus']['root'] = '/om2/user/imgriff/datasets/timit/timit_wsn_compatible.pdpkl'
    
    config['model_name'] = model_name
    config['noise_kwargs']['high_snr'] = snr  
    config['noise_kwargs']['low_snr'] = snr
    config['data']['corpus']['n_talkers'] = num_bg_talkers if not args.get_confusions else False
    config['corpora_name'] = 'TIMIT'
    if snr == 'clean':
        log_name = f"TIMIT_{num_bg_talkers}_talker_{model_name}_{snr}"
    else:
        log_name = f"TIMIT_{num_bg_talkers}_talker_{model_name}_{snr}dB_SNR"

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
        "--eval_cond_file",
        default=pathlib.Path("/om2/user/imgriff/projects/End-to-end-ASR-Pytorch/attn_cnn_n_talker_conds.pkl"),
        type=pathlib.Path,
        help="Directory to save checkpoints and logs to. (Default: './exp')",
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

    args = parser.parse_args()

    run_eval(args)


if __name__ == "__main__":
    cli_main()