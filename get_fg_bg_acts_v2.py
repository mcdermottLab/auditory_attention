import sys 
import yaml
import h5py 
import torch
import pathlib 
import numpy as np 
import argparse
from argparse import ArgumentParser


import src.spatial_attn_lightning as binaural_lightning 
import src.audio_transforms as at
from src.spatial_attn_lightning import BinauralAttentionModule 
from corpus.jsinV3_attn_tracking_multi_talker_background import jsinV3_attn_tracking_multi_talker_background
import src.audio_transforms as at
import torchaudio.transforms as T
from tqdm.auto import tqdm
import pickle

torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch._dynamo.config.skip_nnmodule_hook_guards=False

def get_activations(args):
    print(f"Using attention: {args.attention}")

    # set random seeds 
    torch.manual_seed(0)
    np.random.seed(0)
  
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
    config['data'] = {}
    config['data']['corpus'] = {}
    config['data']['corpus']['n_talkers'] = 1
    config['data']['corpus']['root'] = '/om2/user/msaddler/projects/ibmHearingAid/assets/data/datasets/JSIN_v3.00/nStim_20000/2000ms/rms_0.1/noiseSNR_-10_10/stimSR_20000/reverb_none/noise_all/JSIN_all_v3/subsets/' # Path to raw GigaSpeech dataset
    model_name = pathlib.Path(args.config).stem

    # Set audio transforms  
    snr=0
    audio_transforms = at.AudioCompose([
                    at.AudioToTensor(),
                    at.CombineWithRandomDBSNR(low_snr=-snr, high_snr=snr), 
                    at.RMSNormalizeForegroundAndBackground(rms_level=0.02),  # 0.02 is the default for CV-based models 
                    at.UnsqueezeAudio(dim=0),
                    at.DuplicateChannel() # only need to copy channels here
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
                                              transform=None,
                                              demo=True)

    # set up label re-mapping from SWC to CV labels 
    word_and_speaker_encodings = pickle.load( open( "/om2/user/imgriff/projects/Auditory-Attention/word_and_speaker_encodings_jsinv3.pckl", "rb" )) 
    # key is int, val is word
    wsn_class_map = word_and_speaker_encodings['word_idx_to_word']
    # key is word, val is int
    cv_class_map = pickle.load( open("/om2/user/imgriff/datasets/commonvoice_9/en/cv_800_word_label_to_int_dict.pkl", "rb" )) 
    # map wsn class int key to cv class int value 
    class_remap = {ix:(cv_class_map[word] if word in cv_class_map else -1) for ix, word in wsn_class_map.items()}

    def collate_fn(batch):
        #apply transforsms to batch
        cues = []
        foregrounds = []
        backgrounds = []
        mixtures = []
        labels = []
        for (fg, bg, cue, label) in batch:
            cue = audio_transforms(upsample(torch.from_numpy(cue)).squeeze(), None)[0]
            cues.append(cue)
            fg = upsample(torch.from_numpy(fg)).squeeze()
            bg = upsample(torch.from_numpy(bg)).squeeze()
            foregrounds.append(audio_transforms(fg, None)[0])
            backgrounds.append(audio_transforms(bg, None)[0])
            mixtures.append(audio_transforms(fg, bg)[0])
            labels.append(class_remap[label])

        cues = torch.stack(cues)
        foregrounds = torch.stack(foregrounds)
        backgrounds = torch.stack(backgrounds)
        mixtures = torch.stack(mixtures)
        labels = torch.tensor(labels).type(torch.LongTensor)
        return cues, foregrounds, backgrounds, mixtures, labels

    dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=1,
                num_workers=args.n_jobs,
                collate_fn=collate_fn
            )

    # init array to store activations
    activations = {}

    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output.detach()
        return hook

    n_activations = args.n_activations

    # set hooks for time_average layers & make SigmoidLayers
    conv_modules = {name:module for name, module in model.model._orig_mod.model_dict.named_children() if 'conv' in name or 'relu' in name}
    # add relu fc 
    relu_fc = model.model._orig_mod.relufc
    modules = {**conv_modules, **{'relufc': relu_fc}}
    # dicts to store activations
    cue_reps = {}
    fg_reps = {}
    bg_reps = {}
    mixture_reps = {}

    # register hooks 
    for name, module in modules.items():
        print(name)
        if 'relufc' in name:
            module.register_forward_hook(get_activation(name)) # [2] is relu 
        else:
            module[2].register_forward_hook(get_activation(name)) # [2] is relu 
        # lists to save acts in per-layer
        cue_reps[name] = []
        fg_reps[name] = []
        bg_reps[name] = []
        mixture_reps[name] = []

    # send model to gpu 
    model = model.eval().cuda()

    # add cochleagram to dicts 
    cue_reps['cochleagram'] = []
    mixture_reps['cochleagram'] = []
    fg_reps['cochleagram'] = []
    bg_reps['cochleagram'] = []
    mixture_accs = []
    fg_accs = []
    # get activations 
    n_acts = 0
    with torch.no_grad():
        for ix, batch in tqdm(enumerate(dataloader),  total = n_activations-1):
            fg_cue, foreground, background, mixture, fg_target = batch
            if fg_target == -1:
                continue

            # send to device
            foreground, background, mixture = foreground.cuda(), background.cuda(), mixture.cuda()
            fg_cue =  fg_cue.cuda()

            # convert to cochleagrams
            fg_cue, mixture = coch_gram(fg_cue, mixture)
            foreground, background = coch_gram(foreground, background)
  
            # save inputs
            cue_reps['cochleagram'].append(fg_cue.view(1,-1).cpu())
            mixture_reps['cochleagram'].append(mixture.view(1,-1).cpu())
            fg_reps['cochleagram'].append(foreground.view(1,-1).cpu())
            bg_reps['cochleagram'].append(background.view(1,-1).cpu())

            if not args.attention:
                fg_cue = None 
            
            # run cue only:
            model(None, fg_cue, None)
            for layer in cue_reps.keys():
                if layer == 'cochleagram':
                    continue 
                cue_reps[layer].append(activations[layer].view(1,-1).cpu())
                    
            # run mixture
            preds = model(fg_cue, mixture, None)
            # get model accuracy 
            preds = preds.softmax(-1).argmax(-1).cpu()
            accuracy = (preds == fg_target).int()
            print(f"preds: {preds}, target: {fg_target}, acc: {accuracy}")
            mixture_accs.append(accuracy)
                
            for layer in mixture_reps.keys():
                if layer == 'cochleagram':
                    continue 
                mixture_reps[layer].append(activations[layer].view(1,-1).cpu())
                    
            # run fg
            preds = model(fg_cue, foreground, None)
            # get model accuracy 
            preds = preds.softmax(-1).argmax(-1).cpu()
            accuracy = (preds == fg_target).int()
            fg_accs.append(accuracy)
                
            for layer in fg_reps.keys():
                if layer == 'cochleagram':
                    continue 
                fg_reps[layer].append(activations[layer].view(1,-1).cpu())
                    
            # run bg
            model(fg_cue, background, None)
                
            for layer in bg_reps.keys():
                if layer == 'cochleagram':
                    continue 
                bg_reps[layer].append(activations[layer].view(1,-1).cpu())

            if n_acts == n_activations-1:
                break
            n_acts += 1 
    
    # concat
    cue_reps = {layer:torch.concat(acts,axis=0).numpy().astype('float16')
        for layer,acts in cue_reps.items()}

    mixture_reps = {layer:torch.concat(acts,axis=0).numpy().astype('float16')
        for layer,acts in mixture_reps.items()}

    fg_reps = {layer:torch.concat(acts,axis=0).numpy().astype('float16')
                    for layer,acts in fg_reps.items()}

    bg_reps = {layer:torch.concat(acts,axis=0).numpy().astype('float16')
                    for layer,acts in bg_reps.items()}

    mixture_accs = torch.concat(mixture_accs,axis=0).numpy().astype('int')

    fg_accs = torch.concat(fg_accs,axis=0).numpy().astype('int')
        

    # save activations 
    args.model_dir.mkdir(parents=True, exist_ok=True)
    if not args.attention:
        out_name = args.model_dir / f'{model_name}_model_activations_0dB_no_attention.h5'
    else:
        out_name = args.model_dir / f'{model_name}_model_activations_0dB_w_cues.h5'
    with h5py.File(out_name, 'w') as f:
        for layer in mixture_reps.keys():
            print(f'writing {layer}')
            f.create_dataset(layer+'_cue', data=cue_reps[layer])
            f.create_dataset(layer+'_mixture', data=mixture_reps[layer])
            f.create_dataset(layer+'_fg', data=fg_reps[layer])
            f.create_dataset(layer+'_bg', data=bg_reps[layer])      
        f.create_dataset('acc_mixture', data=mixture_accs)      
        f.create_dataset('acc_fg', data=fg_accs)      



def cli_main():
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default="", help='Path to experiment config.')
    parser.add_argument(
        "--model_dir",
        default=pathlib.Path("./binaural_model_attn_stage_reps"),
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
    # create overwrite flag to handle overwrite of existing results
    parser.add_argument(
        "--attention",
        action=argparse.BooleanOptionalAction,
        help="If true, will skip cue and gains in forward pass",
    )

    args = parser.parse_args()

    get_activations(args)

if __name__ == "__main__":
    cli_main()
