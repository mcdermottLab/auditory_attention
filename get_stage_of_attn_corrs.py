
import numpy as np 
import pickle 
import h5py
from scipy import stats
from tqdm.auto import tqdm
from pathlib import Path 


def get_corrs(args):

    h5_path = Path(args.h5_path)

    with h5py.File(h5_path, 'r') as f:
        layer_names = [key.split("_mixture")[0] for key in f.keys() if "mixture" in key]

       
        # get mixture acts 
        layer = layer_names[args.job_id]
        N_acts = f[f"{layer}_mixture"].shape[0]

        mixture_acts = f[f"{layer}_mixture"]
        target_acts = f[f"{layer}_fg"]
        bg_acts = f[f"{layer}_bg"]

        fg_corr_results = []
        bg_corr_results = []

        for i in range(N_acts):
            fg_corr_results.append(stats.pearsonr(target_acts[i], mixture_acts[i])[0].astype('float32'))
            bg_corr_results.append(stats.pearsonr(bg_acts[i], mixture_acts[i])[0].astype('float32'))

        out_name = f"{h5_path.stem}_corrs.pkl"
        # Save results as dict
        out_name = h5_path.parent / out_name

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
    args = parser.parse_args()

    get_corrs(args)

if __name__ == "__main__":
    cli_main()




