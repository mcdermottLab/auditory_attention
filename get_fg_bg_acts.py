import sys 
import yaml
import torch
import pathlib 
import numpy as np 
from argparse import ArgumentParser

from src.attn_tracking_lightning import AttentionalTrackingModule
from corpus.jsinV3_attn_tracking_multi_talker_background import jsinV3_attn_tracking_multi_talker_background
import src.audio_transforms as at

from tqdm import tqdm

def get_activations(args):
    # Get config for model
    config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
    config['data']['corpus']['n_talkers'] = 1 # want 1 distractor for activations
    
    # Set audio transforms  
    snr_lim = 0
    audio_config = config['data']['audio']

    IIR_COCH = not audio_config['rep_kwargs']['rep_on_gpu']

    if IIR_COCH:
        audio_transforms = at.AudioCompose([
                    at.AudioToTensor(),
                    at.CombineWithRandomDBSNR(low_snr=-snr_lim, high_snr=snr_lim), 
                    at.RMSNormalizeForegroundAndBackground(rms_level=0.1),
                    at.UnsqueezeAudio(dim=0),
                    at.AudioToAudioRepresentation(**audio_config),
            ])
    else:
        audio_transforms = at.AudioCompose([
                    at.AudioToTensor(),
                    at.CombineWithRandomDBSNR(low_snr=-snr_lim, high_snr=snr_lim), 
                    at.RMSNormalizeForegroundAndBackground(rms_level=0.1),
                    at.UnsqueezeAudio(dim=0),
            ])   

    bg_combine_transforms = at.AudioCompose([
            at.AudioToTensor(),
            at.CombineWithRandomDBSNR(low_snr=-snr_lim, high_snr=snr_lim), 
            at.RMSNormalizeForegroundAndBackground(rms_level=0.1),
            at.UnsqueezeAudio(dim=0),
        ])
    cochgram_transforms = at.AudioCompose([
                at.UnsqueezeAudio(dim=0),
                at.AudioToAudioRepresentation(**audio_config),
    ])
    # get model using config
    model = AttentionalTrackingModule(config)

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
    # load checkpoint for model 
    ckpt_path = args.model_dir / args.ckpt

    ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))
    model.load_state_dict(ckpt['state_dict'])

    # init array to store activations
    activations = {}

    def get_activation(name):
        def hook(model, input, output):
            if isinstance(output, tuple):
                cue, mixture = output
                output = mixture
            activations[name] = output.detach()
        return hook

    n_activations = args.n_activations

    # set hooks for time_average layers & make SigmoidLayers
    if IIR_COCH:
        conv_modules = {name:module for name, module in model.model.named_children() if 'conv' in name or 'relufc' in name}
    else:
        get_name = lambda name: 'cochleagram' if name == '0' else 'model' 
        modules = {get_name(name):module for name, module in model.model.named_children()}
        model_modules = modules['model']
        cochleagram = modules['cochleagram']  
        conv_modules = {name:module for name, module in model_modules.named_children() if 'conv' in name or 'relufc' in name}
        conv_modules['cochleagram'] = cochleagram

    # dicts to store activations
    fg_reps = {}
    bg_reps = {}
    mixture_reps = {}

    # register hooks 
    for name, module in conv_modules.items():
        print(name)
        if 'relufc' in name or 'cochleagram' in name:
            module.register_forward_hook(get_activation(name))
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
        for ix, batch in enumerate(tqdm(dataloader, total=n_activations)):
            foreground, background, mixture, fg_cue, fg_target = batch
            
            # convert to cochleagrams
            if IIR_COCH:
                foreground, _ = cochgram_transforms(foreground, None)
                background, _ = cochgram_transforms(background, None)
                foreground = foreground.squeeze(0)
                background = background.squeeze(0)

                # save inputs
                mixture_reps['cochleagram'].append(mixture.view(1,-1))
                fg_reps['cochleagram'].append(foreground.view(1,-1))
                bg_reps['cochleagram'].append(background.view(1,-1))

            # send to device
            foreground, background, mixture = foreground.cuda(), background.cuda(), mixture.cuda()
            fg_cue =  fg_cue.cuda()
            
            # run mixture
            model(fg_cue, mixture)
                
            for layer in mixture_reps.keys():
                if layer == 'cochleagram' and IIR_COCH:
                    continue 
                mixture_reps[layer].append(activations[layer].view(1,-1).cpu())
                    
            # run fg
            model(fg_cue, foreground)
                
            for layer in fg_reps.keys():
                if layer == 'cochleagram' and IIR_COCH:
                    continue 
                fg_reps[layer].append(activations[layer].view(1,-1).cpu())
                    
            # run bg
            model(fg_cue, background)
                
            for layer in bg_reps.keys():
                if layer == 'cochleagram' and IIR_COCH:
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
    out_name = args.model_dir / 'model_activations_0dB.npz'
    np.savez(out_name, mixture_reps=mixture_reps, fg_reps=fg_reps, bg_reps=bg_reps)

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
        "--ckpt",
        default=pathlib.Path("./exp"),
        type=pathlib.Path,
        help="path to checkpoint (Default: './exp')",
    )
    parser.add_argument(
        "--n_activations",
        default=100,
        type=int,
        help="Number of CPUs for dataloader. (Default: 0)",
    )
    parser.add_argument(
        "--n_jobs",
        default=0,
        type=int,
        help="Number of CPUs for dataloader. (Default: 0)",
    )
    args = parser.parse_args()

    get_activations(args)

if __name__ == "__main__":
    cli_main()
