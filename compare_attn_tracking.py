# Following is originally copied from PyTorch RNN-T ASR Example:
# https://github.com/pytorch/audio/tree/820b383b3b21fc06e91631a5b1e6ea1557836216/examples/asr/librispeech_emformer_rnnt

import pathlib
from argparse import ArgumentParser
import yaml

from pytorch_lightning import Trainer, seed_everything
# from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from src.attn_tracking_lightning import AttentionalTrackingModule
from src.attentional_tracking_control_lightning import AttnTrackingControlModule
from src.attn_rove_rms_lightning import AttnRoveRMSModule

# eval_conditions = {
#    0:"Test_Harmonic_Exponential",
#    1:"Test_Harmonic_Gaussian",
#    2:"Test_InharmonicChanging_Exponential",
#    3:"Test_InharmonicChanging_Gaussian",
#    4:"Test_Inharmonic_Exponential",
#    5:"Test_Inharmonic_Gaussian",
#    6:"Test_Interleaved_Exponential",
#    7:"Test_Interleaved_Gaussian"
# }

seed_everything(1)

def run_eval(args):
    config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
    
    config['data']['loader']['num_workers'] = args.n_jobs
    config['data']['loader']['batch_size'] = config['data']['loader']['batch_size'] // args.gpus

    checkpoint_dir = args.exp_dir / "checkpoints"
    print(checkpoint_dir)
    # get latest checkpoint
    checkpoints = sorted(checkpoint_dir.glob('*.ckpt'))
    checkpoint_path = checkpoints[-1] # sort by training step & get latest

    # get eval set & update params 
#     test_condition = eval_conditions[args.array_id]
    experiment_dir = args.exp_dir
    
    #config['data']['corpus']['root'] = f'/om2/user/mjmcp/TestSets/{test_condition}'
    model_name = config['model_name']

    logger = CSVLogger(experiment_dir, name=model_name+"_"+config['snr_condition'])


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
    elif model_name == 'AttnCNN':
        model = AttentionalTrackingModule.load_from_checkpoint(checkpoint_path=checkpoint_path, config=config)  
    elif model_name == "AttnRoveRMSCNN":
        model = AttnRoveRMSModule.load_from_checkpoint(checkpoint_path=checkpoint_path, config=config)
    # evaluate model  
    trainer.test(model)


def cli_main():
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, help='Path to experiment config.')
    parser.add_argument(
        "--exp_dir",
        default=pathlib.Path("./exp"),
        type=pathlib.Path,
        help="Directory to save checkpoints and logs to. (Default: './exp')",
    )
    parser.add_argument(
        "--num_nodes",
        default=1,
        type=int,
        help="Number of nodes to use for training. (Default: 4)",
    )
    parser.add_argument(
        "--gpus",
        default=4,
        type=int,
        help="Number of GPUs per node to use for training. (Default: 8)",
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

    args = parser.parse_args()

    run_eval(args)


if __name__ == "__main__":
    cli_main()