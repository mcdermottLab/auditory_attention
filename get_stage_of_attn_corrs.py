
import numpy as np 
import pickle 
import h5py
from scipy import stats
from tqdm.auto import tqdm
from pathlib import Path 
import argparse
from argparse import ArgumentParser

def get_corrs(args):

    h5_path = Path(args.h5_path)

    if args.run_each_snr:
        h5_files = list(h5_path.glob("*v3*"))
        h5_path = h5_files[args.job_id]

    out_name = f"{h5_path.stem}_corrs.pkl"
    # Save results as dict
    out_name = h5_path.parent / out_name
    
    if out_name.exists():
        return None 
    
    print(f"Getting corrrs for {out_name}")
    
    with h5py.File(h5_path, 'r') as f:
        # layer_names = [key.split("_mixture")[0] for key in f.keys() if "mixture" in key]
        layer_names = [key.split("_mixture")[0] for key in f.keys() if "mixture" in key and 'acc' not in key and 'no_cue' not in key]

        print(layer_names)
        fg_corr_results = {}
        bg_corr_results = {}
        
        N_acts = f[f"{layer_names[0]}_mixture"].shape[0]
        for layer in layer_names:
            mixture_acts = f[f"{layer}_mixture"]
            target_acts = f[f"{layer}_fg"]
            bg_acts = f[f"{layer}_bg"]

            fg_corr_results[layer] = []
            bg_corr_results[layer] = []

            for i in tqdm(range(N_acts), desc=f"Getting activations for {layer}", leave=False):
                fg_corr_results[layer].append(stats.pearsonr(target_acts[i], mixture_acts[i])[0])
                bg_corr_results[layer].append(stats.pearsonr(bg_acts[i], mixture_acts[i])[0])
            
            # set type as float32
            fg_corr_results[layer] = np.array(fg_corr_results[layer], dtype=np.float32)
            bg_corr_results[layer] = np.array(bg_corr_results[layer], dtype=np.float32)
            
        out_dict = dict(fg_corr_results=fg_corr_results, bg_corr_results=bg_corr_results)


        with open(out_name ,'wb') as f:
            pickle.dump(out_dict, f)

def cli_main():
    parser = ArgumentParser()
    parser.add_argument(
        "--h5_path",
        help="Path to h5 file of model activations",
    )
    parser.add_argument(
        "--job_id",
        default=0,
        type=int,
        help="SLURM job array id used to index into config list to select which one to use. (Default: 0)",
        )
    parser.add_argument(
        "--run_each_snr",
        action=argparse.BooleanOptionalAction,
        help="If True, will go through directory and run each set of activations independently.",
    )
    args = parser.parse_args()

    get_corrs(args)

if __name__ == "__main__":
    cli_main()




