import sys 
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
from corpus.swc_mono_test import SWCMonoTestSetH5Dataset
import src.audio_transforms as at
import pandas as pd 
from tqdm.auto import tqdm
import pickle
import soxr

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


## For first pass, alternate whether target or distractor is stationary. 
azims = [0, 10, 350, 45, 315, 90, 270, 180] # azim over 180 is left side 
elevs = [10, -20, 20, 40]
loc_pairs = [(azim, 0) for azim in azims]
loc_pairs += [(0, elev) for elev in elevs]

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
    if time_average:
        activations = activations.mean(dim=-1, keepdim=True)
    if row == 0:
        f.create_dataset(f'{layer}_{suffix}', shape=[n_rows_to_save, np.prod(activations.shape)], dtype=np.float32)
    f[f'{layer}_{suffix}'][row] = activations.cpu().view(-1).numpy()


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
    model_name = Path(args.config).stem

    # Set audio transforms  
    snr = snr_dict[args.job_id]
    audio_transforms_0_db = at.AudioCompose([
                    at.AudioToTensor(),
                    at.BinauralCombineWithRandomDBSNR(low_snr=snr,    # is 0 dB
                                                    high_snr=snr), # is 0 dB 
                    at.BinauralRMSNormalizeForegroundAndBackground(rms_level=0.02), # 20 * np.log10(0.02/20e-6) = 60 dB SPL 
            ])
    audio_transforms_0_db = audio_transforms_0_db.cuda()
    
    # init brir search
    test_IR_manifest_dir = Path(args.room_manifest_path) #  Path("/om2/user/imgriff/spatial_audio_pipeline/assets/brir/mit_bldg46room1004_min_reverb")
    room_ix = args.room_ix
    test_IR_manifest_path = test_IR_manifest_dir / "manifest_brir.pdpkl"
    h5_fn = test_IR_manifest_dir / f"room{room_ix:04}.hdf5"
    new_room_manifest = pd.read_pickle(test_IR_manifest_path)
    only14_manifest = new_room_manifest[(new_room_manifest['index_room'] == room_ix)  & (new_room_manifest['src_dist'] == 1.4)]

    # get latest checkpoint 
    checkpoint_path = args.ckpt_path
    model = BinauralAttentionModule.load_from_checkpoint(checkpoint_path=checkpoint_path, config=config).eval().cuda()
    coch_gram = model.coch_gram.cuda()
    label_type = 'CV'
    sr = config['audio']['rep_kwargs']['sr']
          
    # get dataset
    condition = "one_distractor" # TODO: add logic to run with non-speech distractors 
    dataset = SWCMonoTestSetH5Dataset(h5_path=args.stim_path,
                                    eval_distractor_cond=condition,
                                    model_sr=sr,
                                    label_type=label_type,
                                    for_act_analysis=True)
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

    modules = {name:module for name, module in model.model._orig_mod.model_dict.named_children()}
    # add relu fc 
    relu_fc = model.model._orig_mod.relufc
    modules = {**modules, **{'relufc': relu_fc}}
    # register hooks for all layers 
    for name, module in modules.items():
        if 'conv' in name:
            module[0].register_forward_hook(get_activation(f"{name}_ln")) # [0] is layer norm 
            module[2].register_forward_hook(get_activation(f"{name}_relu")) # [2] is relu 
        else:
            module.register_forward_hook(get_activation(name)) 

    #####################
    # Get gain functions
    ##################### 
    ## Init gain modules per layer. 
    gain_modules = {name:module for name,module in model.model._orig_mod.model_dict.items() if 'attn' in name}
    gain_functions = {}
    for name, module in gain_modules.items():
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
    outname = Path(f'binaural_unit_activations/{model_name}/{model_name}_model_activations_{snr}dB{timg_avg_extn}.h5')
    layer_shape_dict_name = Path(f'binaural_unit_activations/{model_name}/{model_name}_layer_shape_dict{timg_avg_extn}.pkl')
    outname.parent.mkdir(parents=True, exist_ok=True)
    print(f"Preparing to write activations to {outname}")
    # outname = 'test_act_outs.h5'
    n_activations = args.n_activations
    n_rows_to_save = int(n_activations * len(loc_pairs)) # n sounds x n locs 
    layer_shape_dict = {}
    with h5py.File(outname, 'w') as f:
        with torch.no_grad():
            for loc_x, (azim, elev) in enumerate(tqdm(loc_pairs, desc="Location ")):
                target_brir = get_brir(azim=azim, elev=elev, h5_fn=h5_fn, IR_df=only14_manifest, out_sr=sr)
                target_brir = at.Spatialize(target_brir, model_sr=sr, start_crop_in_s=None, end_crop_in_s=None).cuda()
            
                for ix, batch in tqdm(enumerate(dataloader),  total = n_activations, desc=f'Processing activations for location {azim}az {elev}elev', leave=False):
                    ## get row index for saving activations that is global ix of three loops 
                    row = ix + (loc_x * n_activations)
                    # get signals 
                    cue, target, _, _, _, cue_f0, target_f0, _ = batch
                    # spatialize 
                    cue = target_brir(cue.cuda())
                    target = target_brir(target.cuda())
                    # norm and mix transforms 
                    cue, _ = audio_transforms_0_db(cue, None)
                    target, _ = audio_transforms_0_db(target, None)
                    # convert to cochleagrams
                    cue, target = coch_gram(cue, target)
                    # get cochleagram gains - is attn0
                    coch_gains = gain_functions['attn0'](cue)
                
                    if row == 0:
                        f.create_dataset('attncoch_gains', shape=[n_rows_to_save, coch_gains.view(-1).shape[0]], dtype=np.float32)
                        f.create_dataset('cue_f0', shape=[n_rows_to_save], dtype=np.float32)
                        f.create_dataset('target_f0', shape=[n_rows_to_save], dtype=np.float32)
                        f.create_dataset('target_loc', shape=[n_rows_to_save, 2], dtype=np.float32)
                        f.create_dataset('tested_azims', data=azims)
                        f.create_dataset('tested_elevs', data=elevs)
                        
                    # save cochleagram outputs and labels 
                    f['attncoch_gains'][row] = coch_gains.view(-1).cpu().numpy()
                    f['cue_f0'][row] = cue_f0
                    f['target_f0'][row] = target_f0
                    f['target_loc'][row] = [azim, elev]
                    save_activations(f, 'cochleagram', 'cue', cue, row, n_rows_to_save, time_average=args.time_average)
                    save_activations(f, 'cochleagram', 'fg', target, row, n_rows_to_save, time_average=args.time_average)

                    gain_shape_dict = {}
                    model(cue, target, None)  # None is cue_mask_ixs which is not used for activations
                    for layer, acts in activations.items():
                        if 'relufc' in layer or 'attn' in layer:
                            save_activations(f, layer, 'target', acts, row, n_rows_to_save, time_average=args.time_average)
                        else:
                            cue_acts, target_acts = acts
                            save_activations(f, layer, 'cue', cue_acts, row, n_rows_to_save, time_average=args.time_average)
                            save_activations(f, layer, 'target', target_acts, row, n_rows_to_save, time_average=args.time_average)
                            # get gains - these happen before conv block, taking cue from previous pool layer
                            if 'pool' in layer:
                                gain_fn_name = pool_to_gain_map[layer]
                                gain_fn = gain_functions[gain_fn_name]
                                gains = gain_fn(cue_acts)
                                gain_shape_dict[f"{layer}_gains"] = gains.shape
                                save_activations(f, gain_fn_name, 'gains', gains, row, n_rows_to_save)
                            
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
                        default="/om2/user/imgriff/spatial_audio_pipeline/assets/brir/mit_bldg46room1004_min_reverb",
                        type=str, help="Path to room manifest")
    parser.add_argument("--room_ix",
                        default=0,
                        type=int, help="Room index to use") 
    parser.add_argument(
        "--stim_path",
        default=Path("/om/user/imgriff/datasets/human_word_rec_SWC_2023/model_eval_stim.h5"),
        type=Path,
        help="Path where background stimuli are saved",
    )
    parser.add_argument(
        "--time_average",
        action='store_true',
        help="Whether to time average activations",
    )   
    args = parser.parse_args()

    get_activations(args)

if __name__ == "__main__":
    cli_main()
