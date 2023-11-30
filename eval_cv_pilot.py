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
from src.cv_word_lightning import CommonVoiceWordRec
# from src.attentional_tracking_control_lightning import AttnTrackingControlModule
# from src.attn_rove_rms_lightning import AttnRoveRMSModule


seed_everything(1)

def run_eval(args):
    
    model_name = args.model_name
    checkpoint_path = args.ckpt_path
    config = yaml.load(open(args.config_name, 'r'), Loader=yaml.FullLoader)
    
    config['matched_cue_level'] = False

    if args.eval_timit:
        eval_stim_path = '/om2/user/imgriff/datasets/timit/clean_timit_targets_attn_task_0.1rms.pdpkl'
        config['corpora_name'] = 'TIMIT'
        log_name = f"TIMIT_task_clean_{model_name}"

    else:
        eval_stim_path = "/om2/user/imgriff/datasets/commonvoice_9_en/3000ms/stimSR_50000/cv_9_en/subsets/model_and_participant_test_set/model_and_participant_test_set_50000Hz_rate_60dB_000.hdf5"
        config['corpora_name'] = 'model_and_participant_test_set'
        log_name = f"CommonVoice_attn_task_clean_pilot_{model_name}"

    if 'cv' in model_name:
        config['corpus']['root'] = eval_stim_path
        config['loader']['num_workers'] = args.n_jobs
        config['loader']['batch_size'] = 32
    else:
        config['data']['corpus']['root'] = eval_stim_path
        config['data']['loader']['num_workers'] = args.n_jobs
        config['data']['loader']['batch_size'] = 32 # config['data']['loader']['batch_size'] // args.gpus

    snr = 'clean' if args.clean_targets else '0dB_SNR'

    print(log_name)

    experiment_dir = args.exp_dir

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
    elif 'cv' in model_name:
        model = CommonVoiceWordRec.load_from_checkpoint(checkpoint_path=checkpoint_path, config=config)
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
    parser.add_argument(
        "--eval_timit",
        default=False,
        action='store_true',
        help="run timit evaluation",
    )  
    
    args = parser.parse_args()

    run_eval(args)


if __name__ == "__main__":
    cli_main()
