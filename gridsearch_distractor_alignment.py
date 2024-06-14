
import torch 
import numpy as np 
import h5py
from pathlib import Path
from corpus.speaker_room_dataset import SpeakerRoomDataset
import src.audio_transforms as at
import src.spatial_attn_lightning as binaural_lightning 
import yaml
from tqdm.auto import tqdm
from argparse import ArgumentParser

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
    Run gridsearch of distractor signal for a given model and config. 
    Finds onset delay of distractor that that maximizes word classification loss.
    '''
    config_path = Path(args.config_path)
    ckpt_path = Path(args.checkpoint_path)
    # old_style = True 

    config = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)

    config['hparas']['batch_size'] = 2 # config['data']['loader']['batch_size'] // args.gpus
    config['num_workers'] = 0
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
    # audio_transforms = at.AudioCompose([
    #                    at.AudioToTensor(),
    #                    at.BinauralCombineWithRandomDBSNR(low_snr=args.snr,
    #                                                 high_snr=args.snr,
    #                                                 v2_demean=True),
    #                    at.BinauralRMSNormalizeForegroundAndBackground(rms_level=0.02, v2_demean=True), # 20 * np.log10(0.02/20e-6) = 60 dB SPL 
    #             ])

    RANDOMSEED = 1337
    torch.manual_seed(RANDOMSEED)
    np.random.seed(RANDOMSEED)

    clamp_lim = 1.5 # in seconds - is half the signal duration 
    onsets = np.arange(-clamp_lim, clamp_lim, 0.005) # take steps in 5ms increments 
    # convert to samples 
    onsets = (onsets * signal_sr).astype(int)
    coch_transform = model.coch_gram.cuda()

    # save signals as h5, tracking the model name and dataset index
    output_path = Path(args.output_path)
    output_path = output_path / config_path.stem
    output_path.mkdir(parents=True, exist_ok=True)
    # format dataset_ix as 4 digit string
    clamp_lim_ms = int(clamp_lim * 1000)
    output_file = output_path / f"{config_path.stem}_n{clamp_lim_ms}ms_to_p{clamp_lim_ms}ms_5ms_steps.h5"
    print(output_file)

    with h5py.File(output_file, 'w') as f:
        for dataset_ix in tqdm(range(args.n_examples), total=args.n_examples):
            cue, fg, bg, label, _ = dataset[dataset_ix]
            cue, _ = diotic_transforms(cue, None)
            cue = cue.cuda().unsqueeze(0)
            fg = fg.cuda()
            bg = bg.cuda()
            label = torch.tensor(label).unsqueeze(0).cuda()

            loss_fn =  torch.nn.CrossEntropyLoss()

            best_loss = 0.0

            cue, _ = coch_transform(cue, None)    
            ## init h5 if empty 
            if dataset_ix == 0:
                f.create_dataset('original_mixture', shape=[args.n_examples, 2, fg.shape[-1]], dtype=np.float32)
                f.create_dataset('optimized_mixture', shape=[args.n_examples, 2, fg.shape[-1]], dtype=np.float32)
                f.create_dataset('onset_losses', shape=[args.n_examples, onsets.shape[-1]], dtype=np.float32)
                f.create_dataset('best_onsets', shape=[args.n_examples], dtype=np.float32)
                f.create_dataset('onsets_sampled', data=onsets, dtype=np.float32)
            with torch.no_grad():
                for ix, onset in enumerate(tqdm(onsets, leave=False, desc=f"Example {dataset_ix}")):
                    distractor_shifted = torch.roll(bg, onset)
                    mixture_wav, _ = diotic_transforms(fg, distractor_shifted)
                    if ix == 0:
                        f['original_mixture'][dataset_ix, :, :] = prep_torch_to_numpy(mixture_wav)

                    mixture, _ = coch_transform(mixture_wav.unsqueeze(0), None)
                    logits = model(cue, mixture, None)

                    loss = loss_fn(logits, label)
                    f['onset_losses'][dataset_ix, ix] = loss.item()

                    if loss > best_loss:
                        best_loss = loss
                        best_mixture = mixture_wav
                        best_onset = onset / signal_sr

            f['optimized_mixture'][dataset_ix, :, :] = prep_torch_to_numpy(best_mixture)
            f['best_onsets'][dataset_ix] = best_onset

            print(f"Example {dataset_ix}: Best loss of {best_loss:.4f} at {best_onset:.4f}s onset")
        

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str,  help="Path to model checkpoint")
    parser.add_argument("--config_path", type=str, help="Path to model config file")
    parser.add_argument("--output_path", type=str,  help="Path to save signals")
    parser.add_argument("--job_ix", type=int,  default=0, help="Slurm job array ix - used to init random seed")
    parser.add_argument("--n_examples", type=int, default=200, help="Number of examples to run")
    args = parser.parse_args()
    run(args)