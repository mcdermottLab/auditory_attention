
import torch 
import numpy as np 
import pathlib
from pathlib import Path
import yaml
import os
import pickle
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint

import importlib
from src.location_classifier_lightning import LocationClassifier
import argparse
from argparse import ArgumentParser

torch.set_float32_matmul_precision('medium')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


cross_dict = {0: [True, 5000],
              1: [True, 10000],
              2: [False, 5000],
              3: [False, 10000],
              }


def run_train(args):
    seed_everything(123)
    if args.config != "":
        config_path = args.config
    else:
        with open(args.config_list, 'rb') as f:
            model_config = pickle.load(f)
        config_path = model_config[args.job_id]
        config_path = config_path.split("/Auditory-Attention/")[-1]

    config_path = pathlib.Path(config_path)
    config = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)

    with_projection, lim_train_batch = cross_dict[int(args.condition)]
    config['hparas']['lim_train_batch'] = lim_train_batch
    # add transfer learning config
    config['model']['num_classes']['num_locs'] = 504 # 360 azimuth and 90 in elevation
    del config['model']['num_classes']['num_words']
    config['hparas']['batch_size'] = 64
    config['corpus']['task'] = "location"
    config['corpus']['skip_negative_elev'] = True
    n_layers = args.array_id 
    config['model']['n_layers'] = n_layers
    config['model']['with_projection'] = with_projection
    with_projection_str = 'with_projection' if with_projection else 'no_projection'
    config['model']['projection_size'] = 256

    config['hparas']['lr'] = 0.0005
    config['num_workers'] = args.n_jobs
    checkpoint_path = args.ckpt_path
    print(f"Training with config: {config_path.stem}")
    print(f"Using checkpoint: {checkpoint_path}")
    print(f"Using {n_layers} layers")
    model = LocationClassifier(config, ckpt_path=checkpoint_path)
    conv_modules = {name:module for name, module in model.model._orig_mod.model_dict.named_children() if 'conv' in name or 'relu' in name}
    classifier_layer = list(conv_modules.keys())[-1]
    print(f"Using classifier layer: {classifier_layer}")
    callbacks = []
    model_dir = args.exp_dir / config_path.stem / f"{config_path.stem}_{classifier_layer}_{with_projection_str}"
    checkpoint_dir = model_dir / "checkpoints"
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
    print("Init trainer")
    limit = float(args.lim_train_batches)
    if limit > 1:
        limit = int(args.lim_train_batches)
    trainer = Trainer(
        default_root_dir= model_dir,
        precision="32",
        max_epochs=config['hparas']['epochs'],
        num_nodes=1,
        num_sanity_val_steps=2,
        # benchmark=True,
        devices=args.gpus, # was gpus=1,
        val_check_interval=1000,
        gradient_clip_val=100,
        accelerator="gpu",
        profiler=None,
        callbacks=callbacks,
        limit_train_batches = int(args.lim_train_batches),
        limit_val_batches = 50,
    )
    print("training")

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
    parser.add_argument(
        "--array_id",
        default=0,
        type=int,
        help="Slurm array task ID",
    )  
    parser.add_argument('--random_seed', default=0, type=int, help='Random seed for dataset.')
    parser.add_argument('--resume_training', default=False, help='Resume training from checkpoint.')
    parser.add_argument('--negative_elevs', default=False, help='Use negative elevations in training.')
    parser.add_argument('--lim_train_batches', default=1.0, help='Limit the number of training batches.')
    parser.add_argument('--condition', help='key for array condition')
    args = parser.parse_args()

    run_train(args)


if __name__ == "__main__":
    cli_main()
