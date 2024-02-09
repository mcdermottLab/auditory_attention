
import torch 
import numpy as np 
import pathlib
from pathlib import Path
import importlib
import IPython.display as ipd
import src.spatial_attn_lightning as binaural_lightning 
import yaml
import os
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint

import importlib
from src.location_classifier_lightning import LocationClassifier
import argparse
from argparse import ArgumentParser

torch.set_float32_matmul_precision('medium')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def run_train(args):
    seed_everything(123)
    if args.config != "":
        config_path = args.config
    else:
        with open(args.config_list, 'rb') as f:
            model_config = pickle.load(f)
        config_path = model_config[args.job_id]
        config_path = config_path.split("/Auditory-Attention/")[-1]

    print(config_path)
    config_path = pathlib.Path(config_path)
    config = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)

    config['model']['num_classes']['num_locs'] = 504 # 360 azimuth and 90 in elevation
    del config['model']['num_classes']['num_words']
    config['hparas']['batch_size'] = 192

    config['corpus']['task'] = "location"
    config['corpus']['skip_negative_elev'] = True

    config['hparas']['lr'] = 0.0001
    config['num_workers'] = args.n_jobs
    checkpoint_path = args.ckpt_path
    model = LocationClassifier(config, ckpt_path=checkpoint_path)

    callbacks = []

    checkpoint_dir = args.exp_dir / f"{config_path.stem}/checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    callbacks.append(ModelCheckpoint(
            checkpoint_dir,
            monitor=f"{config['val_metric']}",
            mode="max" if 'acc' in config['val_metric'] else "min",
            save_top_k=1,
            save_weights_only=True,
            verbose=True,
        ))
    callbacks.append(ModelCheckpoint(
            checkpoint_dir,
            monitor="train_loss",
            mode="min",
            save_top_k=1,
            save_weights_only=True,
            verbose=True,
        ))

    trainer = Trainer(
        default_root_dir= args.exp_dir / config_path.stem,
        precision="32",
        max_epochs=config['hparas']['epochs'],
        limit_val_batches=0.0,
        num_nodes=1,
        # benchmark=True,
        devices=args.gpus, # was gpus=1,
        val_check_interval=500,
        # gradient_clip_val=100,
        accelerator="gpu",
        profiler=None,
        callbacks=callbacks
    )
    trainer.fit(model)


def cli_main():
    parser = ArgumentParser()
    parser.add_argument('--config', default='', type=str, help='Path to experiment config.')
    parser.add_argument('--config_list', type=str, help='Path to list of config files.')
    parser.add_argument('--job_id', type=int, help='Index into the config list specifying which one to use.')
    parser.add_argument(
        "--exp_dir",
        default=Path("localization_finetune_logs"),
        type=pathlib.Path,
        help="Directory to save checkpoints and logs to. (Default: './exp')",
    )
    parser.add_argument(
        "--ckpt_path",
        default='',
        type=str,
        help="Resume training from this checkpoint."
    )
    parser.add_argument(
        "--num_nodes",
        default=1,
        type=int,
        help="Number of nodes to use for training. (Default: 1)",
    )
    parser.add_argument(
        "--mixed_precision",
        default=True,
        action='store_true',
        help="Use 16 bit precision in training. (Default: False)",
    )
    parser.add_argument(
        "--gpus",
        default=1,
        type=int,
        help="Number of GPUs per node to use for training. (Default: 4)",
    )
    parser.add_argument(
    "--n_jobs",
    default=0,
    type=int,
    help="Number of CPUs for dataloader. (Default: 0)",
    )
    parser.add_argument('--random_seed', default=0, type=int, help='Random seed for dataset.')
    parser.add_argument('--resume_training', default=False, help='Resume training from checkpoint.')
    parser.add_argument('--negative_elevs', default=False, help='Use negative elevations in training.')
    args = parser.parse_args()

    run_train(args)


if __name__ == "__main__":
    cli_main()
