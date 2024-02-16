import sys 
import yaml
import h5py 
import torch
import pathlib 
import numpy as np 
from argparse import ArgumentParser

import src.spatial_attn_lightning as binaural_lightning 
import src.audio_transforms as at
from corpus.binaural_attention_h5 import BinauralAttentionDataset
from corpus.jsinV3_attn_tracking_multi_talker_background import jsinV3_attn_tracking_multi_talker_background
import src.audio_transforms as at
import torchaudio.transforms as T

torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def get_activations(args):
    # Get config for model

    if args.config != "":
        config_path = args.config
    else:
        with open(args.config_list, 'rb') as f:
            model_config = pickle.load(f)
        config_path = model_config[args.job_id]
        config_path = config_path.split("/Auditory-Attention/")[-1]

    print(config_path)
    config = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)
    
    ## update config params for test 
    config['data']['corpus']['n_talkers'] = 1
    config['data']['corpus']['root'] = '/om2/user/msaddler/projects/ibmHearingAid/assets/data/datasets/JSIN_v3.00/nStim_20000/2000ms/rms_0.1/noiseSNR_-10_10/stimSR_20000/reverb_none/noise_all/JSIN_all_v3/subsets/' # Path to raw GigaSpeech dataset

    # Set audio transforms  
    snr=0
    audio_transforms = at.AudioCompose([
                    at.AudioToTensor(),
                    at.CombineWithRandomDBSNR(low_snr=snr, high_snr=snr), 
                    at.RMSNormalizeForegroundAndBackground(rms_level=0.02),  # 0.02 is the default for CV-based models 
                    at.UnsqueezeAudio(dim=0),
            ])  
    
    bg_combine_transforms = at.AudioCompose([
            at.AudioToTensor(),
            at.CombineWithRandomDBSNR(low_snr=snr, high_snr=snr), 
            at.RMSNormalizeForegroundAndBackground(rms_level=0.1),
            at.UnsqueezeAudio(dim=0),
        ])
    

    # get latest checkpoint 
    checkpoint_path = args.ckpt_path
    model = BinauralAttentionModule.load_from_checkpoint(checkpoint_path=checkpoint_path, config=config).eval().cuda()
    coch_gram = model.coch_gram.cuda()
    label_type = 'CV'
    sr = config['audio']['rep_kwargs']['sr']

    # def upsample op
    upsample = T.Resample(20_000, sr,
                        lowpass_filter_width=64,
                        rolloff=0.9475937167399596,
                        beta=14.769656459379492,
                        resampling_method='kaiser_window',
                        dtype=torch.float32)
                                
    # get dataset
    dataset = jsinV3_attn_tracking_multi_talker_background(**config['data']['corpus'],
                                              mode='val',
                                              transform=[audio_transforms, bg_combine_transforms],
                                              demo=True)
    dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=1,
                num_workers=args.n_jobs
            )


    # init array to store activations
    activations = {}

    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output.detach()
        return hook

    n_activations = args.n_activations

    # set hooks for time_average layers & make SigmoidLayers
    conv_modules = {name:module for name, module in model.model.named_children() if 'conv' in name or 'relufc' in name}

    # dicts to store activations
    fg_reps = {}
    bg_reps = {}
    mixture_reps = {}

    # register hooks 
    for name, module in conv_modules.items():
        print(name)
        if 'relufc' in name:
            module.register_forward_hook(get_activation(name)) # [2] is relu 
        else:
            module[2].register_forward_hook(get_activation(name)) # [2] is relu 
        # lists to save acts in per-layer
        fg_reps[name] = []
        bg_reps[name] = []
        mixture_reps[name] = []

    # send model to gpu 
    model = model.eval().cuda()

    # add cochleagram to dicts 
    mixture_reps['cochleagram'] = []
    fg_reps['cochleagram'] = []
    bg_reps['cochleagram'] = []

    # get activations 
    with torch.no_grad():
        for ix, batch in tqdm(enumerate(dataloader),  total = n_activations):
            foreground, background, mixture, fg_cue, fg_target = batch

            # upsample signals 
            fg_cue = upsample(fg_cue)
            foreground = upsample(foreground)
            background = upsample(background)
            mixture = upsample(mixture)
            
            # send to device
            foreground, background, mixture = foreground.cuda(), background.cuda(), mixture.cuda()
            fg_cue =  fg_cue.cuda()

            # convert to cochleagrams
            fg_cue, mixture = coch_gram(fg_cue, mixture)
            foreground, background = coch_gram(foreground, background)

            
  
            # save inputs
            mixture_reps['cochleagram'].append(mixture.view(1,-1).cpu())
            fg_reps['cochleagram'].append(foreground.view(1,-1).cpu())
            bg_reps['cochleagram'].append(background.view(1,-1).cpu())

    
            # run mixture
            model(fg_cue, mixture)
                
            for layer in mixture_reps.keys():
                if layer == 'cochleagram':
                    continue 
                mixture_reps[layer].append(activations[layer].view(1,-1).cpu())
                    
            # run fg
            model(fg_cue, foreground)
                
            for layer in fg_reps.keys():
                if layer == 'cochleagram':
                    continue 
                fg_reps[layer].append(activations[layer].view(1,-1).cpu())
                    
            # run bg
            model(fg_cue, background)
                
            for layer in bg_reps.keys():
                if layer == 'cochleagram':
                    continue 
                bg_reps[layer].append(activations[layer].view(1,-1).cpu())
            
            if ix == n_activations-1:
                break 
        

    # concat and store as numpy arrays
    mixture_reps = {layer:torch.concat(acts,axis=0).numpy().astype('float16')
        for layer,acts in mixture_reps.items()}

    fg_reps = {layer:torch.concat(acts,axis=0).numpy().astype('float16')
                    for layer,acts in fg_reps.items()}

    bg_reps = {layer:torch.concat(acts,axis=0).numpy().astype('float16')
                    for layer,acts in bg_reps.items()}

    # save activations 
    out_name = args.model_dir / 'model_activations_0dB.h5'
    with h5py.File(out_name, 'w') as f:
        for layer in mixture_reps.keys():
            f.create_dataset(layer+'_mixture', data=mixture_reps[layer])
            f.create_dataset(layer+'_fg', data=fg_reps[layer])
            f.create_dataset(layer+'_bg', data=bg_reps[layer])    



def cli_main():
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default="", help='Path to experiment config.')
    parser.add_argument(
        "--model_dir",
        default=pathlib.Path("./exp"),
        type=pathlib.Path,
        help="Directory to save activations to. (Default: './exp')",
    )
    parser.add_argument(
        "--ckpt_path",
        default=pathlib.Path("./exp"),
        type=pathlib.Path,
        help="path to checkpoint (Default: './exp')",
    )
    parser.add_argument(
        "--n_activations",
        default=100,
        type=int,
        help="Number of examples to get activations for. (Default: 100)",
    )
    parser.add_argument(
        "--n_jobs",
        default=0,
        type=int,
        help="Number of CPUs for dataloader. (Default: 0)",
    )
    parser.add_argument(
        "--job_id",
        default=0,
        type=int,
        help="SLURM job array id used to index into config list to select which one to use. (Default: 0)",
        )
    parser.add_argument(
        "--config_list",
        type=str,
        help="Path to dict of config files",
        )
    args = parser.parse_args()

    get_activations(args)

if __name__ == "__main__":
    cli_main()
