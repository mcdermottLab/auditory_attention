### write class to perform ANOVA on unit tuning data for a given newtork layer 
### and save the results as a pkl of a pandas dataframe 

import numpy as np
import pandas as pd
import h5py 
import os
from pathlib import Path 
import statsmodels.api as sm
from statsmodels.formula.api import ols
import argparse
from joblib import Parallel, delayed
from tqdm import tqdm

# import typing 
from typing import List, Tuple, Dict, Any


def optimal_bin_count(data, min_bins=1, max_bins=50):
    # Calculate the Freedman-Diaconis bin width
    q25, q75 = np.percentile(data, [25, 75])
    bin_width = 2 * (q75 - q25) * len(data) ** (-1/3)
    bin_width = max(bin_width, 1e-5)  # Ensure bin width is not zero

    # Calculate the range of bin counts to test
    data_range = np.ptp(data)
    min_bins = max(min_bins, int(np.ceil(data_range / bin_width)))
    max_bins = min(max_bins, int(np.floor(data_range / bin_width)))

    best_bin_count = min_bins
    best_variance = float('inf')

    # Iterate over the range of bin counts
    for bins in range(min_bins, max_bins + 1):
        counts, _ = np.histogram(data, bins=bins)
        variance = np.var(counts)
        if variance < best_variance:
            best_variance = variance
            best_bin_count = bins

    return best_bin_count


def main(args):

    model = args.model_name
    analysis_dir = Path(args.analysis_dir) / model

    h5_fn = analysis_dir / f"{model}_model_activations_0dB_time_avg.h5"

    h5 = h5py.File(h5_fn, 'r')
    print(list(h5.keys()))
    print(list(h5['layer_names']))
    layer_name = f'conv_block_{args.layer_ix}_relu_target'
    target_f0s = h5["target_f0"][:]
    target_locs = h5["target_loc"][:]
    
    ## Filter rows with nan target_f0s
    valid_indices = ~np.isnan(target_f0s)
    target_f0s = target_f0s[valid_indices]
    target_locs = target_locs[valid_indices]

    ## set up labeling 
    unique_locations = np.unique(target_locs, axis=0).astype(int)
    # print(unique_locations)
    location_ixs = {}
    for loc in unique_locations:
        loc_ixs = np.where(np.all(target_locs == loc, axis=1))[0]
        azim, elev = loc
        if elev == 0 and azim == 0:
            location_ixs['front'] = loc_ixs
        elif elev == 0:
            location_ixs[f"{azim} azim"] = loc_ixs
        elif azim == 0:
            location_ixs[f"{elev} elev"] = loc_ixs


    optimal_bins = 15 # optimal_bin_count(target_f0s)
    counts, bins = np.histogram(target_f0s, bins=optimal_bins)
    f0_assignments = np.digitize(target_f0s, bins, right=True)
    bins = bins.round(0)
    f0_bins = bins[f0_assignments]
    f0_bins = f0_bins.astype(int)

    ### init common labeling 
    loc_list = np.zeros(len(target_locs), dtype=object)
    for loc, loc_ixs in location_ixs.items():
        loc_list[loc_ixs] = loc

    f0_list = [f"{bin} Hz" for bin in f0_bins]

    ## make pandas dataframe 
    layer_acts = h5[layer_name][valid_indices]
    n_units = layer_acts.shape[-1]
    n_examples = layer_acts.shape[0]
    dependent_var = layer_acts.flatten()
    unit_ix = np.arange(layer_acts.shape[-1])
    unit_ix = np.tile(unit_ix, n_examples)

    act_df = pd.DataFrame(dependent_var, columns=['activation'])
    act_df['unit_ix'] = unit_ix.astype(int)
    act_df['f0'] = np.repeat(f0_list, n_units)
    act_df['location'] = np.repeat(loc_list, n_units)
    act_df['word_int'] = np.repeat(h5['target_word_int'][valid_indices], n_units).astype('int')
    act_df['speaker_int'] = np.repeat(h5['target_talker_id'][valid_indices], n_units).astype('int')


    formula = 'activation ~ C(f0) + C(location) + C(word_int) + C(speaker_int)' #  + C(f0):C(location) + C(f0):C(word_int) + C(location):C(word_int)'
    prop_var_per_unit = np.zeros((n_units, 4))
    ssq_per_unit = np.zeros((n_units, 4))
    category_labels = ['f0', 'location', 'word', 'speaker'] # , 'f0:location', 'f0:word', 'location:word']

    # Function to process each unit
    def process_unit(unit_i):
        model = ols(formula, act_df[act_df.unit_ix == unit_i]).fit()
        anova_table = sm.stats.anova_lm(model, typ=1)
        total_ss = anova_table.sum_sq.sum()
        ssq = anova_table['sum_sq'][:-1]
        prop_var = ssq / total_ss
        return unit_i, prop_var, ssq

    results = Parallel(n_jobs=args.n_jobs)(delayed(process_unit)(unit_i) for unit_i in tqdm(range(n_units), total=n_units))

    # Collect results
    for unit_i, prop_var, ssq in results:
        prop_var_per_unit[unit_i] = prop_var
        ssq_per_unit[unit_i] = ssq
    
    # Save results
    out_path = analysis_dir 
    out_name = out_path / f"{layer_name}_anova_results.npz"

    # save as npz 
    np.savez(out_name, prop_var_per_unit=prop_var_per_unit, ssq_per_unit=ssq_per_unit, category_labels=category_labels)

    h5.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, help="model name")
    parser.add_argument("--layer_ix", type=int, help="layer index")
    parser.add_argument("--analysis_dir", type=str, help="path to analysis directory")
    parser.add_argument("--n_jobs", type=int, default=1, help="number of jobs to run in parallel")
    args = parser.parse_args()
    main(args)
