# Following is originally copied from PyTorch RNN-T ASR Example:
# https://github.com/pytorch/audio/tree/820b383b3b21fc06e91631a5b1e6ea1557836216/examples/asr/librispeech_emformer_rnnt

import pathlib
from argparse import ArgumentParser
import os
import yaml
import json
import pickle
import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from src.pre_train_lightning import BinauralCNNModule 


torch.set_float32_matmul_precision('medium')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def run_train(args):

    if args.config != "":
        config_path = args.config
        
    else:
        with open(args.config_list, 'rb') as f:
            model_config = pickle.load(f)
        config_path = model_config[args.job_id]
        config_path = config_path.split("/Auditory-Attention/")[-1]

    print(config_path)

    if (config_path.endswith(".json")):
        with open(config_path, 'r') as file:
            config = json.load(file)
    elif (config_path.endswith(".yml")) or (config_path.endswith(".yaml")):
        config = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)
    else:
        print("config file type not supported")
        return


    config['num_workers'] = args.n_jobs
    if args.gpus > 0:
        config['hparas']['batch_size'] = config['hparas']['batch_size'] // args.gpus

    config_path = pathlib.Path(config_path)
    model_name = config_path.stem

    if args.set_lr_with_job:
        print("Setting learning rate with job id")
        model_name = f"{model_name}_lr_1e_neg_{args.job_id+2}"
        config['hparas']['lr'] = 10**(-(args.job_id+2))

    exp_dir = args.exp_dir / model_name
    print("Experiment directory: ", exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = exp_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ckpt_paths = sorted(checkpoint_dir.glob("*.ckpt"), key=os.path.getctime)

    if args.resume_training and len(ckpt_paths) != 0:
        ckpt_path = ckpt_paths[-1]
        seed_everything(int(os.path.getatime(ckpt_path)))
        model = BinauralCNNModule.load_from_checkpoint(checkpoint_path=ckpt_path, config=config)
        print('Resuming training from checkpoint: ', ckpt_path)
    else:
        seed_everything(123)
        model = BinauralCNNModule(config)

    callbacks = []

    if isinstance(config['val_metric'], dict):
        for name, value in config['val_metric'].items():
            callbacks.append(ModelCheckpoint(
                checkpoint_dir,
                filename="{epoch}-{step}-best_"+name,
                monitor=value,
                mode="max",
                save_top_k=1,
                save_weights_only=True,
                verbose=True,
            ))

    else:
        callbacks.append(ModelCheckpoint(
            checkpoint_dir,
            monitor=f"{config['val_metric']}",
            mode="max" if 'acc' in config['val_metric'] else "min",
            save_top_k=1,
            save_weights_only=True,
            verbose=True,
        ))

    train_checkpoint = ModelCheckpoint(
        checkpoint_dir,
        monitor="train_loss",
        mode="min",
        save_top_k=1,
        save_weights_only=True,
        verbose=True,
    )

    callbacks.append(train_checkpoint)

    trainer = Trainer(
        precision="32",
        default_root_dir=exp_dir,
        max_epochs=config['hparas']['epochs'],
        benchmark=True,
        num_nodes=args.num_nodes,
        devices=args.gpus,
        accelerator="gpu", 
        val_check_interval=config['hparas']['valid_step'],
        gradient_clip_val=config['hparas']['gradient_clip_val'],
        profiler=None,
        callbacks=callbacks)

    trainer.fit(model)
    

def cli_main():
    parser = ArgumentParser()
    parser.add_argument('--config', default='config/pre_train_word_loc_task/base_word_loc_pre_train.yaml', type=str, help='Path to experiment config.')
    parser.add_argument('--config_list', type=str, help='Path to list of config files.')
    parser.add_argument('--job_id', type=int, help='Index into the config list specifying which one to use.')
    parser.add_argument(
        "--exp_dir",
        default=pathlib.Path("./exp"),
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
        default=4,
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
    parser.add_argument('--set_lr_with_job', 
                        default=False, 
                        action='store_true',
                        help='Searching for learning rate. Scale with job id')

    args = parser.parse_args()

    run_train(args)


if __name__ == "__main__":
    cli_main()
