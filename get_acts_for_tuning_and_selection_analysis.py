import sys
import os
import yaml
import h5py 
import torch
from pathlib import Path
import numpy as np 
import argparse
from argparse import ArgumentParser
import itertools 
import src.audio_transforms as at
from src.spatial_attn_lightning import BinauralAttentionModule 
from src.spatial_attn_architecture import ECDFGains
from corpus.swc_mono_test import SWCMonoTestSetH5DatasetForUnitTuning
import src.audio_transforms as at
import pandas as pd 
from tqdm.auto import tqdm
import pickle
import soxr
from scipy.stats import pearsonr
from sklearn.metrics.pairwise import cosine_similarity

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



def get_brir(azim=None, elev=None, coords=None, h5_fn=None, IR_df=None, out_sr=44_100):
    if coords is not None:
        azim, elev = coords
    df_row = IR_df[(IR_df['src_azim'] == azim) & (IR_df['src_elev'] == elev)]
    brir_ix = df_row['index_brir'].values[0]
    sr_src = df_row['sr'].values[0]
    with h5py.File(h5_fn, 'r') as f:
        brir = f['brir'][brir_ix]
    if out_sr != sr_src:
        brir = soxr.resample(brir.astype(np.float32), sr_src, out_sr)
    return brir


# need to set up explicit gain module for old attention models 
class AttentionalGains(torch.nn.Module):
    def __init__(self, slope, bias, threshold):
        super(AttentionalGains, self).__init__()
        self.slope = slope.item()
        self.bias = bias.item()
        self.threshold = threshold.item()
        
    def forward(self, cue):
        cue = cue.mean(axis=-1,keepdim=True)
        # apply threshold shift
        cue = cue - self.threshold
        # apply slope
        cue = cue * self.slope
        # apply sigmoid & bias
        gain = self.bias + (1-self.bias) * torch.sigmoid(cue)
        return gain 
    

def save_activations(f, layer, suffix, activations, row, n_rows_to_save, time_average=False):
    """Save activations to the HDF5 file."""
    if time_average and 'relufc' not in layer:
        activations = activations.mean(dim=-1, keepdim=True)
    if row == 0:
        f.create_dataset(f'{layer}_{suffix}', shape=[n_rows_to_save, np.prod(activations.shape)], dtype=np.float32)
    f[f'{layer}_{suffix}'][row] = activations.cpu().view(-1).numpy()


def save_metric(f, layer, suffix, metric, row, n_rows_to_save, is_corr=False):
    """Save activations to the HDF5 file."""
    shape = [n_rows_to_save, 2] if is_corr else [n_rows_to_save]
    if row == 0:
        f.create_dataset(f'{layer}_{suffix}', shape=shape, dtype=np.float32)
    if is_corr:
        f[f'{layer}_{suffix}'][row,:] = metric
    else:
        f[f'{layer}_{suffix}'][row] = metric


def get_activations(args):
    # set random seeds 
    torch.manual_seed(0)
    np.random.seed(0)
  
    # Get config for model
    checkpoint_path = args.ckpt_path
    
    if args.config != "":
        config_path = Path(args.config)
    else:
        with open(args.config_list, 'rb') as f:
            model_config = pickle.load(f)
        config_output = model_config[args.job_id] # is absolute path to model config
        if isinstance(config_output, dict):
            print(f"Loading config from {config_output['config_path']}")
            config_path, checkpoint_path = config_output['config_path'], config_output['ckpt_path']
            config_path = Path(config_path)
        else:
            config_path = Path(config_output)
            
    print(config_path)
    config = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)
    model_name = config_path.stem

    # Set audio transforms  
    # snr = snr_dict[args.job_id]

    snr = 0 
    audio_transforms = at.AudioCompose([
                    at.AudioToTensor(),
                    at.BinauralCombineWithRandomDBSNR(low_snr=snr,    
                                                    high_snr=snr,       
                                                    v2_demean=True), 
                    at.BinauralRMSNormalizeForegroundAndBackground(rms_level=0.02,
                                                                   v2_demean=True), # 20 * np.log10(0.02/20e-6) = 60 dB SPL 
            ])
    audio_transforms = audio_transforms.cuda()
    
    # init brir search
    test_IR_manifest_dir = Path(args.room_manifest_path) #  Path("/om2/user/imgriff/spatial_audio_pipeline/assets/brir/mit_bldg46room1004_min_reverb")
    room_ix = args.room_ix
    test_IR_manifest_path = test_IR_manifest_dir / "manifest_brir.pdpkl"
    h5_fn = test_IR_manifest_dir / f"room{room_ix:04}.hdf5"
    new_room_manifest = pd.read_pickle(test_IR_manifest_path)
    only14_manifest = new_room_manifest[(new_room_manifest['index_room'] == room_ix)  & (new_room_manifest['src_dist'] == 1.4)]

    # handle checkpoint path - if not provided, get latest 
    if checkpoint_path == "":
        ckpt_dir = Path('attn_cue_models/') / model_name / 'checkpoints'
        checkpoint_path = sorted(ckpt_dir.glob("*.ckpt"), key=os.path.getctime)[-1]

    strict_ckpt = True 
    if 'backbone' in model_name:
        config['model']['backbone_with_ecdf_gains'] = True
        strict_ckpt = False
        model_name = f"{model_name}_with_ecdf_gains"

    print(f"Loading {model_name} from {checkpoint_path}")
    ### Set getting acts to true to skip model compile 
    # config['getting_acts'] = True
    rand_weight_str = ""
    if not args.random_weights:
        model = BinauralAttentionModule.load_from_checkpoint(checkpoint_path=checkpoint_path,
                                                             config=config,
                                                             strict=strict_ckpt).eval().cuda()
    else:
        model = BinauralAttentionModule(config=config).eval().cuda()
        rand_weight_str = "_rand_weights"
    coch_gram = model.coch_gram.cuda()
    label_type = 'CV'
    sr = config['audio']['rep_kwargs']['sr']
          
    # get dataset
    dataset = SWCMonoTestSetH5DatasetForUnitTuning(h5_path=args.stim_path,
                                    model_sr=sr,
                               )

    dataloader = torch.utils.data.DataLoader(
                            dataset,
                            batch_size=1,
                            num_workers=args.n_jobs)


    ########################
    # Set hooks for backbone
    ########################
    # init array to store activations
    activations = {}
    def get_activation(name):
        def hook(model, input, output):
            if name in activations:
                activations[name] = torch.cat((activations[name], output.detach()), dim=0)
            else:
                activations[name] = output.detach()
        return hook

    if hasattr(model.model, '_orig_mod'):
        modules = {name:module for name, module in model.model._orig_mod.model_dict.named_children()}
        relu_fc = model.model._orig_mod.relufc

    else:
        modules = {name:module for name, module in model.model.model_dict.named_children()}
        relu_fc = model.model.relufc

    # add relu fc 
    modules = {**modules, **{'relufc': relu_fc}}
    # register hooks for all layers 
    for name, module in modules.items():
        if 'conv' in name:
            module[0].register_forward_hook(get_activation(f"{name}_ln")) # [0] is layer norm 
            module[2].register_forward_hook(get_activation(f"{name}_relu")) # [2] is relu 
        if 'ecdf' in model_name and 'attn' in name:
            continue
        else:
            module.register_forward_hook(get_activation(name)) 

    #####################
    # Get gain functions
    ##################### 
    ## Init gain modules per layer. 
    
    if hasattr(model.model, '_orig_mod'):
        gain_modules = {name:module for name,module in model.model._orig_mod.model_dict.items() if 'attn' in name}
    else:
        gain_modules = {name:module for name,module in model.model.model_dict.items() if 'attn' in name}
   
    gain_functions = {} 
    for name, module in gain_modules.items():
        if 'backbone' in model_name and 'ecdf' in model_name:
            gain_functions[name] = module
        else:
            gain_functions[name] = AttentionalGains(module.slope, module.bias, module.threshold)

    ##################################################################
    # Set dict mapping layer names to their corresponding gain modules
    ##################################################################
    # want dict that maps conv layer name 'conv_block_<int>_hannpool' to attn module 'attn<int>'
    ## Map pool layers to gain functions 
    pool_layers = [name for name in modules.keys() if "pool" in name]
    n_pool_layers = len(pool_layers)
    pool_to_gain_map = {}
    pool_to_gain_map['cochleagram'] = 'attn0'
    for ix, layer in enumerate(pool_layers):
        if str(n_pool_layers - 1) in layer:
            pool_to_gain_map[layer] = 'attnfc'
        else:
            pool_to_gain_map[layer] = f"attn{ix+1}"

    # send model to gpu 
    model = model.eval().cuda()
    # get activations 
    if args.time_average:
        timg_avg_extn = '_time_avg'
    else:
        timg_avg_extn = ''

    if args.center_loc_only:
        center_loc_str = '_center_loc_only'
    else:
        center_loc_str = ''

    if 'main' in model_name:
        model_name = model_name + "_latest_ckpt"

    if args.cue_single_source:
        cue_single_source_str = '_cue_single_source'
    else:
        cue_single_source_str = ''
    if args.diotic:
        outname = Path(f'binaural_unit_activation_analysis/{model_name}{rand_weight_str}/{model_name}{rand_weight_str}_model_activations_{snr}dB{timg_avg_extn}_diotic{cue_single_source_str}.h5')
    else:
        outname = Path(f'binaural_unit_activation_analysis/{model_name}{rand_weight_str}/{model_name}{rand_weight_str}_model_activations_{snr}dB{timg_avg_extn}{center_loc_str}{cue_single_source_str}.h5')

    out_dir = Path("/om/scratch/Thu/imgriff")
    outname = out_dir / outname 

    layer_shape_dict_name = Path(f'binaural_unit_activation_analysis/{model_name}/{model_name}_layer_shape_dict{timg_avg_extn}.pkl')
    layer_shape_dict_name.parent.mkdir(parents=True, exist_ok=True)
    outname.parent.mkdir(parents=True, exist_ok=True)
    print(f"Preparing to write activations to {outname}")
    # outname = 'test_act_outs.h5'
    n_activations = args.n_activations
    layer_shape_dict = {}

    ## For first pass, alternate whether target or distractor is stationary. 
    azims = [0, 10, 350, 45, 315, 90, 270, 180] # azim over 180 is left side 
    elevs = [10, -20, 20, 40]
    loc_pairs = [(azim, 0) for azim in azims]
    loc_pairs += [(0, elev) for elev in elevs]

    if outname.exists() and not args.overwrite:
        print(f"{outname} already exists. Exiting.")
        sys.exit()

    with h5py.File(outname, 'w') as f:
        with torch.no_grad():
            if args.diotic or args.center_loc_only:
                loc_pairs = [(0, 0)]
            n_rows_to_save = int(n_activations * len(loc_pairs)) # n sounds x n locs 
            for loc_x, (azim, elev) in enumerate(tqdm(loc_pairs, desc="Location ")):
                target_brir = get_brir(azim=azim, elev=elev, h5_fn=h5_fn, IR_df=only14_manifest, out_sr=sr)
                target_brir = at.Spatialize(target_brir, model_sr=sr, start_crop_in_s=None, end_crop_in_s=None).cuda()
            
                for ix, batch in tqdm(enumerate(dataloader),  total = n_activations, desc=f'Processing activations for location {azim}az {elev}elev', leave=False):
   
                    ## get row index for saving activations that is global ix of three loops 
                    row = ix + (loc_x * n_activations)
                    # get signals 
                    cue, target, same_sex_dist, diff_sex_dist, nat_scene_dist, word_int, target_f0, same_dist_f0, diff_dist_f0 = batch
                    # spatialize 
                    if args.diotic:
                        batch_size = cue.shape[0]
                        # copy channels 
                        cue = cue.repeat(2, 1).unsqueeze(0).cuda()
                        target = target.repeat(2, 1).unsqueeze(0).cuda()
                        same_sex_dist = same_sex_dist.repeat(2, 1).unsqueeze(0).cuda()
                        diff_sex_dist = diff_sex_dist.repeat(2, 1).unsqueeze(0).cuda()
                        nat_scene_dist = nat_scene_dist.repeat(2, 1).unsqueeze(0).cuda()
                    else:
                        cue = target_brir(cue.cuda())
                        target = target_brir(target.cuda())
                        same_sex_dist = target_brir(same_sex_dist.cuda())
                        diff_sex_dist = target_brir(diff_sex_dist.cuda())
                        nat_scene_dist = target_brir(nat_scene_dist.cuda())
                    # norm and mix transforms 
                    cue, _ = audio_transforms(cue, None)
                    target, _ = audio_transforms(target, None)
                    same_sex_dist, _ = audio_transforms(same_sex_dist, None)
                    diff_sex_dist, _ = audio_transforms(diff_sex_dist, None)
                    nat_scene_dist, _ = audio_transforms(nat_scene_dist, None)

                    if row == 0:
                        # only need to do this once 
                        silence_cue = torch.zeros_like(cue, device='cuda')
                        silence_cue, _ = coch_gram(silence_cue, None)

            
                    # get mixture signals 
                    mixture_same, _ = audio_transforms(target, same_sex_dist)
                    mixture_diff, _ = audio_transforms(target, diff_sex_dist)
                    mixture_nat_scene, _ = audio_transforms(target, nat_scene_dist)
    
                    # convert to cochleagrams
                    cue, target = coch_gram(cue, target)
                    same_sex_dist, diff_sex_dist = coch_gram(same_sex_dist, diff_sex_dist)
                    mixture_same, mixture_diff = coch_gram(mixture_same, mixture_diff)
                    nat_scene_dist, mixture_nat_scene = coch_gram(nat_scene_dist, mixture_nat_scene)

                    # if args.cue_single_source:
                    #     single_source_cue = cue 
                    # else:
                    #     single_source_cue = silence_cue
                        
                    if not ('control' in config_path.stem or 'late_only' in config_path.stem):
                        # get cochleagram gains - is attn0
                        coch_gains = gain_functions['attn0'](cue)
                
                    if row == 0:
                        if not ('control' in config_path.stem or 'late_only' in config_path.stem):
                            f.create_dataset('attncoch_gains', shape=[n_rows_to_save, coch_gains.view(-1).shape[0]], dtype=np.float32)
                        f.create_dataset('target_f0', shape=[n_rows_to_save], dtype=np.float32)
                        f.create_dataset('same_dist_f0', shape=[n_rows_to_save], dtype=np.float32)
                        f.create_dataset('diff_dist_f0', shape=[n_rows_to_save], dtype=np.float32)
                        f.create_dataset('target_word_int', shape=[n_rows_to_save], dtype=np.float32)
                        f.create_dataset('target_loc', shape=[n_rows_to_save, 2], dtype=np.float32)
                        f.create_dataset('tested_azims', data=azims)
                        f.create_dataset('tested_elevs', data=elevs)
                        
                    # save cochleagram outputs and labels 
                    if not ('control' in config_path.stem or 'late_only' in config_path.stem):
                        f['attncoch_gains'][row] = coch_gains.view(-1).cpu().numpy()
                    f['target_f0'][row] = target_f0
                    f['same_dist_f0'][row] = same_dist_f0
                    f['diff_dist_f0'][row] = diff_dist_f0
                    f['target_word_int'][row] = word_int
                    f['target_loc'][row] = [azim, elev]

                    save_activations(f, 'cochleagram', 'cue', cue, row, n_rows_to_save, time_average=args.time_average)
                    save_activations(f, 'cochleagram', 'target', target, row, n_rows_to_save, time_average=args.time_average)
                    save_activations(f, 'cochleagram', 'same_sex_dist', same_sex_dist, row, n_rows_to_save, time_average=args.time_average)
                    save_activations(f, 'cochleagram', 'diff_sex_dist', diff_sex_dist, row, n_rows_to_save, time_average=args.time_average)
                    save_activations(f, 'cochleagram', 'nat_scene_dist', nat_scene_dist, row, n_rows_to_save, time_average=args.time_average)
                    save_activations(f, 'cochleagram', 'mixture_same', mixture_same, row, n_rows_to_save, time_average=args.time_average)
                    save_activations(f, 'cochleagram', 'mixture_diff', mixture_diff, row, n_rows_to_save, time_average=args.time_average)
                    save_activations(f, 'cochleagram', 'mixture_nat_scene', mixture_nat_scene, row, n_rows_to_save, time_average=args.time_average)

                    ## Corr  between fg and each mixture 
                    corr_same = pearsonr(target.view(-1).cpu().numpy(), mixture_same.view(-1).cpu().numpy())
                    save_metric(f, 'cochleagram', 'target_mixture_same_corr', corr_same, row, n_rows_to_save, is_corr=True)
                    corr_diff = pearsonr(target.view(-1).cpu().numpy(), mixture_diff.view(-1).cpu().numpy())
                    save_metric(f, 'cochleagram', 'target_mixture_diff_corr', corr_diff, row, n_rows_to_save, is_corr=True)
                    corr_nat_scene = pearsonr(target.view(-1).cpu().numpy(), mixture_nat_scene.view(-1).cpu().numpy())
                    save_metric(f, 'cochleagram', 'target_mixture_nat_scene_corr', corr_nat_scene, row, n_rows_to_save, is_corr=True)
                    ## Corr between each distractor and corresponding mixture 
                    corr_same_mix_same = pearsonr(same_sex_dist.view(-1).cpu().numpy(), mixture_same.view(-1).cpu().numpy())
                    save_metric(f, 'cochleagram', 'same_dist_mixture_same_corr', corr_same_mix_same, row, n_rows_to_save, is_corr=True)
                    corr_diff_mix_diff = pearsonr(diff_sex_dist.view(-1).cpu().numpy(), mixture_diff.view(-1).cpu().numpy())
                    save_metric(f, 'cochleagram', 'diff_dist_mixture_diff_corr', corr_diff_mix_diff, row, n_rows_to_save, is_corr=True)
                    corr_nat_scene_mix_nat_scene = pearsonr(nat_scene_dist.view(-1).cpu().numpy(), mixture_nat_scene.view(-1).cpu().numpy())
                    save_metric(f, 'cochleagram', 'nat_scene_dist_mixture_nat_scene_corr', corr_nat_scene_mix_nat_scene, row, n_rows_to_save, is_corr=True)


                    # get activations per layer 
                    for dis_str, mixture in zip(['same', 'diff', 'nat_scene'], [mixture_same, mixture_diff, mixture_nat_scene]):
                        activations = {}
                        gain_shape_dict = {}
                        model(cue, mixture, None)  # None is cue_mask_ixs which is not used for activations
                        for layer, acts in activations.items():
                            if 'relufc' in layer or 'attn' in layer or 'control' in config_path.stem:
                                save_activations(f, layer, f"mixture_{dis_str}", acts, row, n_rows_to_save, time_average=args.time_average)
                            else:
                                cue_acts, mixture_acts = acts
                                    # cue activations will be same for all mixtures 
                                save_activations(f, layer, f"cue_{dis_str}", cue_acts, row, n_rows_to_save, time_average=args.time_average) 
                                save_activations(f, layer, f"mixture_{dis_str}", mixture_acts, row, n_rows_to_save, time_average=args.time_average)
                                # get gains - these happen before conv block, taking cue from previous pool layer
                                if 'pool' in layer and dis_str == 'same': # only need todo this once 
                                    gain_fn_name = pool_to_gain_map[layer]
                                    if gain_fn_name in gain_functions:
                                        gain_fn = gain_functions[gain_fn_name]
                                        gains = gain_fn(cue_acts)
                                        gain_shape_dict[f"{layer}_gains"] = gains.shape
                                        save_activations(f, gain_fn_name, 'gains', gains, row, n_rows_to_save)
                    
                    ## Process single source signals and get corrs 
                    for source_str, source in zip(['target', 'same_sex_dist', 'diff_sex_dist', 'nat_scene_dist'], [target, same_sex_dist, diff_sex_dist, nat_scene_dist]):
                        # run without cue 
                        activations = {}
                        model(silence_cue, source, None)
                        for layer, acts in activations.items():
                            if len(acts) == 2:
                                _, acts = acts
                            save_activations(f, layer, source_str, acts, row, n_rows_to_save, time_average=args.time_average)
                            # for cpr 
                            if 'relufc' in layer or not args.time_average:
                                acts = acts.cpu().view(-1).numpy()
                            else:
                                acts = acts.mean(-1).cpu().view(-1).numpy()
                            # get corrs between target and each source
                            if source_str == 'target':
                                for mixture_str in ["mixture_same", "mixture_diff", "mixture_nat_scene"]:
                                    mixture_acts = f[f"{layer}_{mixture_str}"][row]
                                    corr = pearsonr(acts, mixture_acts)
                                    save_metric(f, layer, f"{source_str}_{mixture_str}_corr", corr, row, n_rows_to_save, is_corr=True)
                            elif source_str == 'same_sex_dist':
                                corr = pearsonr(acts, f[f"{layer}_mixture_same"][row])
                                save_metric(f, layer, f"{source_str}_mixture_same_corr", corr, row, n_rows_to_save, is_corr=True)
                            elif source_str == 'diff_sex_dist':
                                corr = pearsonr(acts, f[f"{layer}_mixture_diff"][row])
                                save_metric(f, layer, f"{source_str}_mixture_diff_corr", corr, row, n_rows_to_save, is_corr=True)
                            elif source_str == 'nat_scene_dist':
                                corr = pearsonr(acts, f[f"{layer}_mixture_nat_scene"][row])
                                save_metric(f, layer, f"{source_str}_mixture_nat_scene_corr", corr, row, n_rows_to_save, is_corr=True)
                        
                        # run with cue 
                        activations = {}
                        model(cue, source, None)
                        for layer, acts in activations.items():
                            if len(acts) == 2:
                                _, acts = acts
                            save_activations(f, layer, f"cued_{source_str}", acts, row, n_rows_to_save, time_average=args.time_average)
                            # for cpr 
                            if 'relufc' in layer or not args.time_average:
                                acts = acts.cpu().view(-1).numpy()
                            else:
                                acts = acts.mean(-1).cpu().view(-1).numpy()
                            # get corrs between target and each source
                            if source_str == 'target':
                                for mixture_str in ["mixture_same", "mixture_diff", "mixture_nat_scene"]:
                                    mixture_acts = f[f"{layer}_{mixture_str}"][row]
                                    corr = pearsonr(acts, mixture_acts)
                                    save_metric(f, layer, f"cued_{source_str}_{mixture_str}_corr", corr, row, n_rows_to_save, is_corr=True)
                            elif source_str == 'same_sex_dist':
                                corr = pearsonr(acts, f[f"{layer}_mixture_same"][row])
                                save_metric(f, layer, f"cued_{source_str}_mixture_same_corr", corr, row, n_rows_to_save, is_corr=True)
                            elif source_str == 'diff_sex_dist':
                                corr = pearsonr(acts, f[f"{layer}_mixture_diff"][row])
                                save_metric(f, layer, f"cued_{source_str}_mixture_diff_corr", corr, row, n_rows_to_save, is_corr=True)
                            elif source_str == 'nat_scene_dist':
                                corr = pearsonr(acts, f[f"{layer}_mixture_nat_scene"][row])
                                save_metric(f, layer, f"cued_{source_str}_mixture_nat_scene_corr", corr, row, n_rows_to_save, is_corr=True)

                    if row == 0:
                        layer_shape_dict = {layer: activations[layer].shape for layer in activations.keys()}
                        shape_dict = {**layer_shape_dict, **gain_shape_dict}
                        with open(layer_shape_dict_name, 'wb') as p:
                            pickle.dump(shape_dict, p)
                        layer_names = [name.encode('utf-8') for name in activations.keys()]
                        f.create_dataset('layer_names', data=layer_names)


                    # reset to clear memory 
                    activations={}
                    if ix == n_activations-1:
                        break
                

def cli_main():
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default="", help='Path to experiment config.')
    parser.add_argument(
        "--model_dir",
        default=Path("./binaural_unit_tuning"),
        type=Path,
        help="Directory to save activations to. (Default: './exp')",
    )
    parser.add_argument(
        "--ckpt_path",
        default=Path("./exp"),
        type=Path,
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
    parser.add_argument("--room_manifest_path", 
                        default="/om2/user/msaddler/spatial_audio_pipeline/assets/brir/eval",
                        type=str,
                        help="Path to room manifest")
    parser.add_argument("--room_ix",
                        default=0,
                        type=int,
                        help="Room index to use") 
    parser.add_argument(
        "--stim_path",
        default=Path("/om/user/imgriff/datasets/human_word_rec_SWC_2024/model_eval_stim.h5"),
        type=Path,
        help="Path where background stimuli are saved",
    )
    parser.add_argument(
        "--time_average",
        action='store_true',
        help="Whether to time average activations",
    )  
    parser.add_argument(
        "--diotic",
        action='store_true',
        help="Whether to run without spatialization",
    )   
    parser.add_argument(
        "--center_loc_only",
        action='store_true',
        help="Whether to run only at center location.",
    )
    parser.add_argument(
        "--random_weights",
        action='store_true',
        help="Whether to run using random weights",
    ) 
    parser.add_argument(
        "--cue_single_source",
        action='store_true',
        help="Whether to use the cue signal or silence when getting activations for single sources. True uses cue, False uses silence. Default is False.",
    )
    parser.add_argument(
        "--overwrite",
        action='store_true',
        help="Whether to overwrite existing activations file.",
    )
    args = parser.parse_args()

    get_activations(args)

if __name__ == "__main__":
    cli_main()
