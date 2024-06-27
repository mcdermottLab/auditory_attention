import sys 
import yaml
import h5py 
import torch
import pathlib 
from pathlib import Path
import numpy as np 
import argparse
from argparse import ArgumentParser
from scipy.stats import pearsonr
from sklearn.metrics.pairwise import cosine_similarity


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

snr_dict = {2: -8,
            1: -5, 
            0: 0,
            3: 5,
            4: 8
}

def get_activations(args):

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
    config['data']['corpus']['root'] = '/om/user/imgriff/datasets/dataset_word_speaker_noise/JSIN_all_v3/subsets/' # New path to JSIN dataset
    model_name = pathlib.Path(args.config).stem

    # Set audio transforms  
    snr = snr_dict[args.job_id]
    audio_transforms = at.AudioCompose([
                    at.AudioToTensor(),
                    at.CombineWithRandomDBSNR(low_snr=snr, high_snr=snr), 
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
            # print(name)
            if name in activations:
                activations[name] = torch.cat((activations[name], output.detach()), dim=0)
            else:
                activations[name] = output.detach()
        return hook

    n_activations = args.n_activations

    # set hooks for time_average layers & make SigmoidLayers
    conv_modules = {name:module for name, module in model.model._orig_mod.model_dict.named_children() if 'conv' in name or 'relu' in name}
    # add relu fc 
    relu_fc = model.model._orig_mod.relufc
    modules = {**conv_modules, **{'relufc': relu_fc}}

    # register hooks 
    for name, module in modules.items():
        print(name)
        if 'relufc' in name:
            module.register_forward_hook(get_activation(name)) 
        else:
            module[2].register_forward_hook(get_activation(name)) # [2] is relu 

    # send model to gpu 
    model = model.eval().cuda()
    # get activations 
    

    outname = Path(f'binaural_model_attn_stage_reps/{model_name}/{model_name}_model_activations_{snr}dB_w_cues_and_corrs_v3.h5')
    outname.parent.mkdir(parents=True, exist_ok=True)
    print(f"Preparing to write activations to {outname}")
    # outname = 'test_act_outs.h5'
    with h5py.File(outname, 'w') as f:

        with torch.no_grad():
            for ix, batch in tqdm(enumerate(dataloader),  total = n_activations):
                fg_cue, foreground, background, mixture, fg_target = batch
                if ix == 0:
                    if args.silence_w_uncued:
                        uncued_cue = torch.zeros_like(fg_cue, device='cuda')
                        uncued_cue, _ = coch_gram(uncued_cue, None)
                    else:
                        uncued_cue = None
                # send to device
                foreground, background, mixture = foreground.cuda(), background.cuda(), mixture.cuda()
                fg_cue =  fg_cue.cuda()

                # convert to cochleagrams
                fg_cue, mixture = coch_gram(fg_cue, mixture)
                foreground, background = coch_gram(foreground, background)

                if ix == 0:
                    f.create_dataset('cochleagram_cue',shape=[n_activations, mixture.view(-1).shape[0]], dtype=np.float32)
                    f.create_dataset('cochleagram_mixture', shape=[n_activations, mixture.view(-1).shape[0]], dtype=np.float32)
                    f.create_dataset('cochleagram_fg', shape=[n_activations, mixture.view(-1).shape[0]], dtype=np.float32)
                    f.create_dataset('cochleagram_bg', shape=[n_activations, mixture.view(-1).shape[0]], dtype=np.float32)
                    f.create_dataset('cochleagram_cue_mixture_corr', shape=[n_activations, 2], dtype=np.float32)
                    f.create_dataset('cochleagram_fg_mixture_corr', shape=[n_activations, 2], dtype=np.float32)
                    f.create_dataset('cochleagram_bg_mixture_corr', shape=[n_activations, 2], dtype=np.float32)
                    f.create_dataset('cochleagram_cue_mixture_cos', shape=[n_activations], dtype=np.float32)
                    f.create_dataset('cochleagram_fg_mixture_cos', shape=[n_activations], dtype=np.float32)
                    f.create_dataset('cochleagram_bg_mixture_cos', shape=[n_activations], dtype=np.float32)
                f['cochleagram_cue'][ix] = fg_cue.view(-1).cpu().numpy()
                f['cochleagram_mixture'][ix] = mixture.view(-1).cpu().numpy()
                f['cochleagram_fg'][ix] = foreground.view(-1).cpu().numpy()
                f['cochleagram_bg'][ix] = background.view(-1).cpu().numpy()
                # get cors 
                f['cochleagram_cue_mixture_corr'][ix,:] = pearsonr(fg_cue.view(-1).cpu().numpy(), mixture.view(-1).cpu().numpy())
                f['cochleagram_fg_mixture_corr'][ix,:] = pearsonr(foreground.view(-1).cpu().numpy(), mixture.view(-1).cpu().numpy())
                f['cochleagram_bg_mixture_corr'][ix,:] = pearsonr(background.view(-1).cpu().numpy(), mixture.view(-1).cpu().numpy())
                # get cos sim
                f['cochleagram_cue_mixture_cos'][ix] = cosine_similarity(fg_cue.view(1,-1).cpu().numpy(), mixture.view(1,-1).cpu().numpy())
                f['cochleagram_fg_mixture_cos'][ix] = cosine_similarity(foreground.view(1,-1).cpu().numpy(), mixture.view(1,-1).cpu().numpy())
                f['cochleagram_bg_mixture_cos'][ix] = cosine_similarity(background.view(1,-1).cpu().numpy(), mixture.view(1,-1).cpu().numpy())
                
                # run mixture
                pred = model(fg_cue, mixture, None)
                for layer in activations.keys():
                    if 'relufc' in layer:
                        mixture_acts = activations[layer]
                        mixture_acts = mixture_acts.cpu() 
                        if ix == 0:
                            f.create_dataset(f'{layer}_mixture', shape=[n_activations, np.prod(mixture_acts.shape)], dtype=np.float32)
                    else:
                        cue_acts, mixture_acts = activations[layer] 
                        cue_acts, mixture_acts = cue_acts.cpu(), mixture_acts.cpu()
                        if ix == 0:
                            f.create_dataset(f'{layer}_cue', shape=[n_activations, np.prod(cue_acts.shape)], dtype=np.float32)
                            f.create_dataset(f'{layer}_mixture', shape=[n_activations, np.prod(mixture_acts.shape)], dtype=np.float32)
                            f.create_dataset(f'{layer}_cue_mixture_corr', shape=[n_activations, 2], dtype=np.float32)
                            f.create_dataset(f'{layer}_cue_mixture_cos', shape=[n_activations], dtype=np.float32)
                        f[f'{layer}_cue'][ix] = cue_acts.view(-1).numpy()
                        f[f'{layer}_cue_mixture_corr'][ix,:] = pearsonr(cue_acts.view(-1).numpy(), mixture_acts.view(-1).numpy())
                        f[f'{layer}_cue_mixture_cos'][ix] = cosine_similarity(cue_acts.view(1,-1).numpy(), mixture_acts.view(1,-1).numpy())
                    f[f'{layer}_mixture'][ix] = mixture_acts.view(-1).numpy()
                activations = {}

                # run fg - can skip cue 
                pred = model(fg_cue, foreground, None)
                for layer in activations.keys():
                    if 'relufc' in layer:
                        fg_acts = activations[layer] 
                    else:
                        _, fg_acts = activations[layer]
                    if ix == 0:
                        f.create_dataset(f'{layer}_fg', shape=[n_activations, np.prod(fg_acts.shape)], dtype=np.float32)
                        f.create_dataset(f'{layer}_fg_mixture_corr', shape=[n_activations, 2], dtype=np.float32)
                        f.create_dataset(f'{layer}_fg_mixture_cos', shape=[n_activations], dtype=np.float32)
                    fg_acts = fg_acts.cpu()
                    f[f'{layer}_fg_mixture_corr'][ix,:] = pearsonr(fg_acts.view(-1).numpy(), f[f'{layer}_mixture'][ix])
                    f[f'{layer}_fg_mixture_cos'][ix] = cosine_similarity(fg_acts.view(1,-1).numpy(), f[f'{layer}_mixture'][ix].reshape(1,-1))
                    f[f'{layer}_fg'][ix] = fg_acts.view(-1).numpy()
                activations = {}

                # run bg
                pred = model(fg_cue, background, None)
                for layer in activations.keys():
                    if 'relufc' in layer:
                        bg_acts = activations[layer]
                    else:
                        _, bg_acts = activations[layer]
                    if ix == 0:
                        # get flattened shape of bg_acts 
                        f.create_dataset(f'{layer}_bg', shape=[n_activations, np.prod(bg_acts.shape)], dtype=np.float32)
                        f.create_dataset(f'{layer}_bg_mixture_corr', shape=[n_activations, 2], dtype=np.float32)
                        f.create_dataset(f'{layer}_bg_mixture_cos', shape=[n_activations], dtype=np.float32)
                    bg_acts = bg_acts.cpu()
                    f[f'{layer}_bg_mixture_corr'][ix,:] = pearsonr(bg_acts.view(-1).numpy(),  f[f'{layer}_mixture'][ix])
                    f[f'{layer}_bg_mixture_cos'][ix] = cosine_similarity(bg_acts.view(1,-1).numpy(), f[f'{layer}_mixture'][ix].reshape(1,-1))
                    f[f'{layer}_bg'][ix] = bg_acts.view(-1).numpy()
                activations = {}
                

                # Save signals without cue in forward pass 
                # run mixture
                pred = model(uncued_cue, mixture, None)
                for layer in activations.keys():
                    if 'relufc' in layer:
                        uncued_mixture_acts = activations[layer] 
                    else:
                        _, uncued_mixture_acts = activations[layer]
                    if ix == 0:
                        f.create_dataset(f'{layer}_mixture_no_cue', shape=[n_activations, np.prod(uncued_mixture_acts.shape)], dtype=np.float32)
                    uncued_mixture_acts = uncued_mixture_acts.cpu()
                    f[f'{layer}_mixture_no_cue'][ix] = uncued_mixture_acts.view(-1).numpy()
                activations = {}

                # run fg - can skip cue
                pred = model(uncued_cue, foreground, None)
                for layer in activations.keys():
                    if 'relufc' in layer:
                        fg_acts = activations[layer] 
                    else:
                        _, fg_acts = activations[layer]
                    fg_acts = fg_acts.cpu()
                    if ix == 0:
                        f.create_dataset(f'{layer}_fg_no_cue', shape=[n_activations, np.prod(fg_acts.shape)], dtype=np.float32)
                        f.create_dataset(f'{layer}_fg_mixture_corr_no_cue', shape=[n_activations, 2], dtype=np.float32)
                        f.create_dataset(f'{layer}_fg_mixture_cos_no_cue', shape=[n_activations], dtype=np.float32)
                    f[f'{layer}_fg_no_cue'][ix] = fg_acts.view(-1).numpy()
                    f[f'{layer}_fg_mixture_corr_no_cue'][ix,:] = pearsonr(fg_acts.view(-1).numpy(), f[f'{layer}_mixture'][ix])
                    f[f'{layer}_fg_mixture_cos_no_cue'][ix] = cosine_similarity(fg_acts.view(1,-1).numpy(), f[f'{layer}_mixture'][ix].reshape(1,-1))
                activations = {}

                # run bg
                pred = model(uncued_cue, background, None)
                for layer in activations.keys():
                    if 'relufc' in layer:
                        bg_acts = activations[layer] 
                    else:
                        _, bg_acts = activations[layer]
                    bg_acts = bg_acts.cpu()
                    if ix == 0:
                        f.create_dataset(f'{layer}_bg_no_cue', shape=[n_activations, np.prod(bg_acts.shape)], dtype=np.float32)
                        f.create_dataset(f'{layer}_bg_mixture_corr_no_cue', shape=[n_activations, 2], dtype=np.float32)
                        f.create_dataset(f'{layer}_bg_mixture_cos_no_cue', shape=[n_activations], dtype=np.float32)
                    f[f'{layer}_bg_no_cue'][ix] = bg_acts.view(-1).numpy()
                    f[f'{layer}_bg_mixture_corr_no_cue'][ix,:] = pearsonr(bg_acts.view(-1).numpy(), f[f'{layer}_mixture'][ix])
                    f[f'{layer}_bg_mixture_cos_no_cue'][ix] = cosine_similarity(bg_acts.view(1,-1).numpy(), f[f'{layer}_mixture'][ix].reshape(1,-1))
                activations = {}

                if ix == n_activations-1:
                    break
        

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
        "--silence_w_uncued",
        action=argparse.BooleanOptionalAction,
        help="If True, use silence in uncued trials. (Default: False)",
    )

    args = parser.parse_args()

    get_activations(args)

if __name__ == "__main__":
    cli_main()
