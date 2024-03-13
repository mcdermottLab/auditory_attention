
import torch 
import numpy as np 
import h5py
import os
import sys
from pathlib import Path
from corpus.speaker_room_dataset import SpeakerRoomDataset
import src.audio_transforms as at
import src.spatial_attn_lightning as binaural_lightning 
import yaml
from tqdm.auto import tqdm
import src.spatial_attn_architecture as attn_arch
import re
import src.custom_modules as cm
from argparse import ArgumentParser
from torch.optim.lr_scheduler import OneCycleLR

torch.set_float32_matmul_precision('medium')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def prep_torch_to_numpy(torch_tensor):
    if torch_tensor.is_cuda:
        torch_tensor = torch_tensor.cpu()
    if torch_tensor.requires_grad:
        torch_tensor = torch_tensor.detach()

    return torch_tensor.squeeze().numpy()


def run(args):
    '''
    Run optimization of distractor signal for a given model and config. 
    Generates distractor signal that maximizes word classification loss.
    '''
    config_path = Path(args.config_path)
    ckpt_path = Path(args.checkpoint_path)
    # old_style = True 

    config = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)

    config['hparas']['batch_size'] = 2 # config['data']['loader']['batch_size'] // args.gpus
    config['num_workers'] = 2
    config['noise_kwargs']['low_snr'] = 0
    config['noise_kwargs']['high_snr'] = 0
    # get model input sr for brir resampling
    signal_sr = config['audio']['rep_kwargs']['sr']
    coch_sr = config['audio']['rep_kwargs']['env_sr']


    model = binaural_lightning.BinauralAttentionModule.load_from_checkpoint(checkpoint_path=ckpt_path, config=config).cuda().eval()


    dataset = SpeakerRoomDataset(manifest_path='/om2/user/rphess/Auditory-Attention/final_binaural_manifest.pkl',
                                excerpt_path='/om2/user/msaddler/spatial_audio_pipeline/assets/swc/manifest_all_words.pdpkl',
                                cue_type='voice_and_location',
                                sr=signal_sr) 


    diotic_transforms = at.AudioCompose([
                        at.AudioToTensor(),
                        at.CombineWithRandomDBSNR(low_snr=0, high_snr=0), 
                        at.RMSNormalizeForegroundAndBackground(rms_level=0.02),  # 0.02 is the default for CV-based models 
                        at.DuplicateChannel(),
                ])
    diotic_transforms = diotic_transforms.cuda()
    
    # for target + distractor combination 
    audio_transforms = at.AudioCompose([
                       at.AudioToTensor(),
                       at.BinauralCombineWithRandomDBSNR(low_snr=0,
                                                    high_snr=0,
                                                    v2_demean=True),
                       at.BinauralRMSNormalizeForegroundAndBackground(rms_level=0.02, v2_demean=True), # 20 * np.log10(0.02/20e-6) = 60 dB SPL 
                ])


    RANDOMSEED = args.job_ix
    NOISESCALE = args.noise_scale
    EARLYSTOP = args.early_stop
    torch.manual_seed(RANDOMSEED)
    np.random.seed(RANDOMSEED)

    audio_transforms = model.audio_transforms.cuda()
    coch_transform = model.coch_gram.cuda()

    dataset_ix =  args.job_ix 
    cue, fg, bg, label, confusion = dataset[dataset_ix]

    cue, _ = diotic_transforms(cue, None)
    fg, _ = diotic_transforms(fg, None)
    # push to gpu 
    cue = cue.cuda().unsqueeze(0)
    fg = fg.cuda().unsqueeze(0)
    label = torch.tensor(label).unsqueeze(0).cuda()

    # init as noise 
    distractor = (torch.rand_like(fg) * NOISESCALE).cuda()
    init_distractor = distractor.detach().clone().cpu()
    distractor.requires_grad_(True)
    distractor.retain_grad()

    optimizer = torch.optim.SGD(
                [distractor],
                lr=args.learning_rate,
                momentum=0.9,
                weight_decay=5e-4,
            )

    loss_fn =  torch.nn.CrossEntropyLoss()

    n_steps = args.n_steps

    best_loss = 0
    best_distractor = None
    best_step = 0 
    early_stop = EARLYSTOP

    cue_cg = None 
    for step in (pbar := tqdm(range(n_steps))):
        optimizer.zero_grad()
        mixture, _ = audio_transforms(fg, distractor)
        if cue_cg == None:
            cue_cg, mixture = coch_transform(cue, mixture)
        else:
            mixture, _ = coch_transform(mixture, None)
        logits = model(cue_cg, mixture, None)
        loss = -loss_fn(logits, label)

        if loss < best_loss:
            best_loss = loss.detach().item()
            best_distractor = distractor.detach().clone()
            best_step = step 
            early_stop = EARLYSTOP
            # add best loss to tqdm progress bar 
            pbar.set_postfix_str(f"Best loss: {best_loss:.3f} Current loss: {loss.detach().item():.3f} Early stop: {early_stop}")
        else:
            early_stop -= 1 

        if step % 100 == 0: 
            pbar.set_postfix_str(f"Best loss: {best_loss:.3f} Current loss: {loss.detach().item():.3f} Early stop: {early_stop}")

        if early_stop == 0:
            break 

        loss.backward()
        optimizer.step()

    print(f"Best final loss: {best_loss.detach().item():.3f}")
    # 
    # save signals as h5, tracking the model name and dataset index
    output_path = Path(args.output_path)
    output_path = output_path / config_path.stem
    output_path.mkdir(parents=True, exist_ok=True)
    # format dataset_ix as 4 digit string
    output_file = output_path / f"{config_path.stem}_distractor_{dataset_ix:04d}.h5"
    # convert torch tensors to numpy 
    cue = prep_torch_to_numpy(cue)
    fg = prep_torch_to_numpy(fg)
    best_distractor = prep_torch_to_numpy(best_distractor)
    init_distractor = prep_torch_to_numpy(init_distractor)
    # save signals as h5, tracking the model name and dataset index
    with h5py.File(output_file, 'w') as f:
        # save cue, target, initial distractor, best distractor, best step, and best loss
        f.create_dataset('target', data=fg, dtype=np.float32) 
        f.create_dataset('cue', data=cue, dtype=np.float32)
        f.create_dataset('init_distractor', data=init_distractor, dtype=np.float32)
        f.create_dataset('best_distractor', data=best_distractor, dtype=np.float32)
        f.create_dataset('best_step', data=np.array([best_step]), dtype=np.int32)
        f.create_dataset('best_loss', data=np.array([best_loss.detach().item()]), dtype=np.float32)
        f.create_dataset('dataset_ix', data=np.array([dataset_ix]), dtype=np.int32)
        f.create_dataset('learning_rate', data=np.array([args.learning_rate]), dtype=np.float32)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str,  help="Path to model checkpoint")
    parser.add_argument("--config_path", type=str, help="Path to model config file")
    parser.add_argument("--output_path", type=str,  help="Path to save signals")
    parser.add_argument("--job_ix", type=int,  default=0, help="Slurm job array ix - used to init random seed")
    parser.add_argument("--noise_scale", type=float,  default=0.000001, help="Initial noise variance for distractor optimization")
    parser.add_argument("--early_stop", type=int,  default=1000, help="Number of steps before early stopping")
    parser.add_argument("--n_steps", type=int,  default=10000, help="Number of steps for optimization")
    parser.add_argument("--learning_rate", type=float,  default=0.1, help="Learning rate for optimization")
    args = parser.parse_args()
    run(args)