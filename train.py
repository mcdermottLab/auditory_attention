# Following is originally copied from PyTorch RNN-T ASR Example:
# https://github.com/pytorch/audio/tree/820b383b3b21fc06e91631a5b1e6ea1557836216/examples/asr/librispeech_emformer_rnnt

import pathlib
from argparse import ArgumentParser
import yaml
import json

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.plugins import DDPPlugin

# get nodename 
import socket


hostname = socket.gethostname()


def run_train(args):
    if (args.config.endswith(".json")):
        with open(args.config, 'r') as file:
            config = json.load(file)
    elif (args.config.endswith(".yaml")):
        config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
    else:
        print("config file type not supported")
        print(args.config)
        return
      
    config['n_jobs'] = args.n_jobs
    if args.gpus > 0:
        config['data']['loader']['batch_size'] = config['data']['loader']['batch_size'] // args.gpus
    else:
        config['data']['loader']['batch_size'] = 1
    
    checkpoint_dir = args.exp_dir / "checkpoints"
    if args.ckpt_path != '':
        ckpt_path = checkpoint_dir / args.ckpt_path
    else:
        ckpt_path = None
        
    # if  'dgx002' in hostname:
    #     config['data']['corpus']['root'] = '/mnt/local-scratch/JSIN_v3.00'
        
    callbacks = []
    
    if isinstance(config['val_metric'], dict):
        for name, value in config['val_metric'].items():
            callbacks.append(ModelCheckpoint(
                checkpoint_dir,
                filename="{epoch}-{step}-best_"+name,
                monitor=value,
                mode="max",
                save_top_k=1,
#                 save_weights_only=True,
                verbose=True,
            ))
    
    else:
        callbacks.append(ModelCheckpoint(
            checkpoint_dir,
            monitor=f"{config['val_metric']}",
            mode="max" if 'ACC' in config['val_metric'] else "min",
            save_top_k=1,
#             save_weights_only=True,
            verbose=True,
        ))
        
    train_checkpoint = ModelCheckpoint(
        checkpoint_dir,
        monitor="Losses/train_loss",
        mode="min",
        save_top_k=1,
#         save_weights_only=True,
        verbose=True,
    )
    
    callbacks.append(train_checkpoint)
   
    trainer = Trainer(
        # precision=16 if args.mixed_precision else 32,
        precision=16,
        default_root_dir=args.exp_dir,
        max_epochs=config['hparas']['epochs'],
    
       # log_every_n_steps = 10,
        detect_anomaly=False,
        num_nodes=args.num_nodes,
        gpus=args.gpus,
        accelerator="gpu" if args.gpus > 0 else 'cpu',
        # resume_from_checkpoint = ckpt_path,  
        strategy=DDPPlugin(find_unused_parameters=False),
        val_check_interval=config['hparas']['valid_step'],
        gradient_clip_val=config['hparas']['gradient_clip_val'],
        profiler=None,
        callbacks=callbacks)

    if 'commonvoice' in args.config.as_posix():
        from src.cv_word_lightning import CommonVoiceWordRec
        print('CommonVoice Task')
        module = CommonVoiceWordRec

    else:
        from src.attn_tracking_lightning import AttentionalTrackingModule
        module = AttentionalTrackingModule
    
    if ckpt_path:
        model =  module.load_from_checkpoint(checkpoint_path=ckpt_path, config=config) 
    else:
        model = module(config)
    
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
    parser.add_argument(
        "--ckpt_path",
        default='',
        type=str,
        help="Resume training from this checkpoint."
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
        help="Number of nodes to use for training. (Default: 1)",
    )
    parser.add_argument(
        "--mixed_precision",
        default=True,
        action='store_true',
        help="Use 16 bit precision in training. (Default: False)",
    )
    parser.add_argument(
        "--dgx002_path",
        default=False,
        action='store_true',
        help="use dgx002 jsin dataset",
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
    args = parser.parse_args()

    run_train(args)


if __name__ == "__main__":
    cli_main()
