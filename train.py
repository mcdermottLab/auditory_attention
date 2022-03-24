# Following is originally copied from PyTorch RNN-T ASR Example:
# https://github.com/pytorch/audio/tree/820b383b3b21fc06e91631a5b1e6ea1557836216/examples/asr/librispeech_emformer_rnnt

import pathlib
from argparse import ArgumentParser
import yaml

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint


def run_train(args):
    config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
    checkpoint_dir = args.exp_dir / "checkpoints"
    checkpoint = ModelCheckpoint(
        checkpoint_dir,
        monitor="WER/val_att",
        mode="min",
        save_top_k=1,
        save_weights_only=True,
        verbose=True,
    )
    train_checkpoint = ModelCheckpoint(
        checkpoint_dir,
        monitor="Losses/train_loss",
        mode="min",
        save_top_k=1,
        save_weights_only=True,
        verbose=True,
    )
    callbacks = [
        checkpoint,
        train_checkpoint,
    ]
    trainer = Trainer(
        default_root_dir=args.exp_dir,
        max_epochs=config['hparas']['epochs'],
        num_nodes=args.num_nodes,
        gpus=args.gpus,
        accelerator="gpu",
        strategy="ddp",
        val_check_interval=config['hparas']['valid_step'],
        gradient_clip_val=10.0,
        callbacks=callbacks,
    )
    if config['model_name'] == 'RNNT':
        from src.giga_rnnt_lightning import RNNTModule
        model = RNNTModule(config)
    elif config['model_name'] == 'LAS':
        from src.giga_las_lightning import LASModule
        model = LASModule(config)      
    trainer.fit(model)


def cli_main():
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, help='Path to experiment config.')
    parser.add_argument(
        "--exp_dir",
        default=pathlib.Path("./exp"),
        type=pathlib.Path,
        help="Directory to save checkpoints and logs to. (Default: './exp')",
    )
    # parser.add_argument(
    #     "--global_stats_path",
    #     default=pathlib.Path("global_stats.json"),
    #     type=pathlib.Path,
    #     help="Path to JSON file containing feature means and stddevs.",
    # )
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
    args = parser.parse_args()

    run_train(args)


if __name__ == "__main__":
    cli_main()
