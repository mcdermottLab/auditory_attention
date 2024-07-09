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
azims = [0, 10, 45, 90, 180]
elevs = [0, 10, 40]
loc_pairs = list(itertools.product(*[azims, elevs]))

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
    
def save_activations(f, layer, suffix, activations, row, n_rows_to_save):
    """Save activations to the HDF5 file."""
    if row == 0:
        f.create_dataset(f'{layer}_{suffix}', shape=[n_rows_to_save, np.prod(activations.shape)], dtype=np.float32)
    f[f'{layer}_{suffix}'][row] = activations.cpu().view(-1).numpy()

def process_and_save_activations(model, cue, target, condition_suffix, row, n_rows_to_save, f, save_cue=False):
    """Process and save activations for a given condition."""
    activations = {}  # Assuming this gets filled somewhere within model call or globally accessible
    attended_acts = {}
    model(cue, target, None)  # Assuming the third parameter is always None for simplicity
    for layer in activations.keys():
        if 'relufc' in layer:
            target_acts = activations[layer]
            save_activations(f, layer, condition_suffix, target_acts, row, n_rows_to_save)
        else:
            cue_acts, target_acts = activations[layer]
            if save_cue:
                save_activations(f, layer, condition_suffix, cue_acts, row, n_rows_to_save)
            save_activations(f, layer, condition_suffix, target_acts, row, n_rows_to_save)
    for layer in attended_acts.keys():
        save_activations(f, layer, condition_suffix, attended_acts[layer], row, n_rows_to_save)
        
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
            # print(name)
            if name in activations:
                activations[name] = torch.cat((activations[name], output.detach()), dim=0)
            else:
                activations[name] = output.detach()
        return hook

    conv_modules = {name:module for name, module in model.model._orig_mod.model_dict.named_children() if 'conv' in name or 'relu' in name}
    # add relu fc 
    relu_fc = model.model._orig_mod.relufc
    modules = {**conv_modules, **{'relufc': relu_fc}}
    # register hooks for all layers 
    for name, module in modules.items():
        print(name)
        if 'relufc' in name:
            module.register_forward_hook(get_activation(name)) 
        else:
            # save all stages of CNN - may need to softcode this layer for new architectures is ok for v07
            module[0].register_forward_hook(get_activation(f"{name}_ln")) # [0] is layer norm 
            module[2].register_forward_hook(get_activation(f"{name}_relu")) # [2] is relu 
            module[3].register_forward_hook(get_activation(f"{name}_hannpool")) # [3] is pool 

    #########################
    # Set hooks for attention
    #########################
    ## Init gain modules per layer. 
    gain_modules = {name:module for name,module in model.model.model_dict.items() if 'attn' in name}
    gain_functions = {}

    for name, module in gain_modules.items():
        gain_functions[name] = AttentionalGains(module.slope, module.bias, module.threshold)

    attended_acts = {}
    def get_attention(name):
        def hook(model, input, output):
            # print(name)
            if name in attended_acts:
                attended_acts[name] = torch.cat((attended_acts[name], output.detach()), dim=0)
            else:
                attended_acts[name] = output.detach()
        return hook

    # register hooks 
    for name, module in gain_modules.items():
            module.register_forward_hook(get_attention(name)) 


        # send model to gpu 
    model = model.eval().cuda()
    # get activations 
    
    outname = Path(f'binaural_unit_activations/{model_name}/{model_name}_model_activations_{snr}dB.h5')
    outname.parent.mkdir(parents=True, exist_ok=True)
    print(f"Preparing to write activations to {outname}")
    # outname = 'test_act_outs.h5'
    n_activations = args.n_activations
    n_rows_to_save = int(n_activations * len(loc_pairs) * 2) # n sounds x n locs x target vs distractor moving 
    with h5py.File(outname, 'w') as f:
        with torch.no_grad():
            for loc_x, loc in enumerate(loc_pairs):
                for sig_x, loc_str in enumerate(['target', 'distractor']):
                    if loc_str == 'target':
                        # target is moving:
                        target_loc = loc,
                        distractor_loc = [0,0]
                    else:
                        # distractor is moving:
                        target_loc = [0,0]
                        distractor_loc = loc

                    target_brir = at.get_brir(azim=target_loc[0], elev=target_loc[1], h5_fn=h5_fn, IR_df=only14_manifest, out_sr=sr)
                    target_brir = at.Spatialize(target_brir, model_sr=sr).cuda()
                    
                    distractor_brir = at.get_brir(azim=distractor_loc[0], elev=distractor_loc[1] ,h5_fn=h5_fn, IR_df=only14_manifest, out_sr=sr)
                    distractor_brir = at.Spatialize(distractor_brir, model_sr=sr).cuda()
                   
                    for ix, batch in tqdm(enumerate(dataloader),  total = n_activations):
                        ## get row index for saving activations that is global ix of three loops 
                        row = ix + sig_x * n_activations + loc_x * (n_activations * 2)
                        # get signals 
                        cue, target, distractor, _, _, cue_f0, target_f0, dist_f0 = batch
                        # spatialize 
                        cue = target_brir(cue.cuda())
                        target = target_brir(target.cuda())
                        distractor = distractor_brir(distractor.cuda())
                        # norm and mix transforms 
                        cue, _ = audio_transforms_0_db(cue, None)
                        mixture, _ = audio_transforms_0_db(target, distractor)
                        # do target and distractor after mixutre for compat 
                        target, _ = audio_transforms_0_db(target, None)
                        distractor, _ = audio_transforms_0_db(distractor, None)

                        # spatialize signals
                        if row == 0:
                            silence_cue = torch.zeros_like(cue, device='cuda')
                            silence_cue, _ = coch_gram(silence_cue, None)

                        if row == 0:
                            f.create_dataset('cochleagram_cue',shape=[n_rows_to_save, mixture.view(-1).shape[0]], dtype=np.float32)
                            f.create_dataset('cochleagram_mixture', shape=[n_rows_to_save, mixture.view(-1).shape[0]], dtype=np.float32)
                            f.create_dataset('cochleagram_fg', shape=[n_rows_to_save, mixture.view(-1).shape[0]], dtype=np.float32)
                            f.create_dataset('cochleagram_bg', shape=[n_rows_to_save, mixture.view(-1).shape[0]], dtype=np.float32)
                            f.create_dataset('cue_f0', shape=[n_rows_to_save], dtype=np.float32)
                            f.create_dataset('target_f0', shape=[n_rows_to_save], dtype=np.float32)
                            f.create_dataset('distractor_f0', shape=[n_rows_to_save], dtype=np.float32)
                            f.create_dataset('target_loc', shape=[n_rows_to_save, 2], dtype=np.float32)
                            f.create_dataset('distractor_loc', shape=[n_rows_to_save, 2], dtype=np.float32)
      
                        # convert to cochleagrams
                        cue, mixture = coch_gram(cue, mixture)
                        target, distractor = coch_gram(target, distractor)

                        # save cochleagram outputs and labels 
                        f['cochleagram_cue'][row] = cue.view(-1).cpu().numpy()
                        f['cochleagram_mixture'][row] = mixture.view(-1).cpu().numpy()
                        f['cochleagram_fg'][row] = target.view(-1).cpu().numpy()
                        f['cochleagram_bg'][row] = distractor.view(-1).cpu().numpy()
                        f['cue_f0'][row] = cue_f0
                        f['target_f0'][row] = target_f0
                        f['distractor_f0'][row] = dist_f0
                        f['target_loc'][row] = target_loc
                        f['distractor_loc'][row] = distractor_loc
                    
                        # run mixture
                        process_and_save_activations(model, cue, mixture, 'mixture', row, n_rows_to_save, f, save_cue=True)
                    
                        # run fg - can skip saving cue 
                        process_and_save_activations(model, cue, target, 'fg', row, n_rows_to_save, f, save_cue=False)

                        # run bg - can skip saving cue 
                        process_and_save_activations(model, cue, distractor, 'bg', row, n_rows_to_save, f, save_cue=False)
                        
                        # Save signals when uncued (silence_cue) in forward pass 
                        # run mixture
                        process_and_save_activations(model, silence_cue, mixture, 'mixture_no_cue', row, n_rows_to_save, f, save_cue=False)

                        # run fg - can skip cue
                        process_and_save_activations(model, silence_cue, target, 'fg_no_cue', row, n_rows_to_save, f, save_cue=False)

                        # run bg
                        process_and_save_activations(model, silence_cue, distractor, 'distractor_no_cue', row, n_rows_to_save, f, save_cue=False)

                        if row == n_activations-1:
                            break
                

def cli_main():
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default="", help='Path to experiment config.')
    parser.add_argument(
        "--model_dir",
        default=Path("./binaural_model_attn_stage_reps"),
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
