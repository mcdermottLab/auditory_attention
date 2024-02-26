
import numpy as np 
import pickle 
import h5py
from scipy import stats
from tqdm.auto import tqdm
from pathlib import Path 
from argparse import ArgumentParser
from joblib import Parallel, delayed


def get_corrs(args):
    np.random.seed(args.job_id)

    h5_path = Path(args.h5_path)
    model_name = h5_path.parent.stem

    with h5py.File(h5_path, 'r', swmr=True) as f:
        layer_names = [key.split("_mixture")[0] for key in f.keys() if "mixture" in key]

       
        # get mixture acts 
        layer = layer_names[args.job_id]
        N_acts = f[f"{layer}_mixture"].shape[0]

        n_boot = args.n_boot 

        mixture_acts = f[f"{layer}_mixture"][:]
        target_acts = f[f"{layer}_fg"][:]

        # correlate each act against all other acts
        # parallelize this
        def calculate_correlation(target_acts, mixture_acts):
            i,j = np.random.choice(N_acts, size=2, replace=False)
            fg_fg_corr = stats.pearsonr(target_acts[i], target_acts[j])[0]
            fg_mix_corr = stats.pearsonr(target_acts[i], mixture_acts[j])[0]
            return (fg_fg_corr, fg_mix_corr)

        results = Parallel(n_jobs=args.n_jobs)(delayed(calculate_correlation)(target_acts, mixture_acts) for _ in tqdm(range(n_boot), total=n_boot, desc=f"Getting null activations for {layer}"))

        layer_fg_corrs, layer_mixture_corrs = zip(*results)
        layer_fg_corrs = np.array(layer_fg_corrs, dtype=np.float32)
        layer_mixture_corrs = np.array(layer_mixture_corrs, dtype=np.float32)

        layer_fg_corrs = {'mean': np.nanmean(layer_fg_corrs),
                                 'two_sem': 2*np.nanstd(layer_fg_corrs)/np.sqrt(n_boot)}
        layer_mixture_corrs = {'mean': np.nanmean(layer_mixture_corrs),
                                 'two_sem': 2*np.nanstd(layer_mixture_corrs)/np.sqrt(n_boot)}
        
        out_name = f"{h5_path.stem}_{layer}_null_corrs.pkl"
        # Save results as dict
        out_name = h5_path.parent / out_name

        out_dict = dict(null_fg_fg_corr_results=layer_fg_corrs,
                        null_fg_mixture_corr_results=layer_mixture_corrs)

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
        "--n_boot",
        default=10_000,
        type=int,
        help="Number of bootstrap iterations to draw. (Default: 10_000)",
        )
    parser.add_argument(
        "--n_jobs",
        default=1,
        type=int,
        help="Number of parallel jobs. (Default: 1)",
        )
    args = parser.parse_args()

    get_corrs(args)

if __name__ == "__main__":
    cli_main()




