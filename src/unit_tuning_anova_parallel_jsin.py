### write class to perform ANOVA on unit tuning data for a given network layer 
### and save the results as a pkl of a pandas dataframe 
### Modified to support SLURM job arrays with chunked unit processing

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


def get_layer_info(h5, layer_names):
    """Get number of units for each layer"""
    layer_info = []
    for layer_name in layer_names:
        try:
            layer_shape = h5[layer_name].shape
            n_units = layer_shape[-1]
            layer_info.append({'name': layer_name, 'n_units': n_units})
            print(f"  {layer_name}: {n_units} units")
        except Exception as e:
            print(f"Could not access layer {layer_name}: {e}")
    return layer_info


def compute_layer_and_unit_indices(job_array_idx, layer_info, units_per_job):
    """
    Compute which layer and which unit range to process based on job array index.
    
    Returns:
        tuple: (layer_idx, start_unit, end_unit, layer_name, n_units_in_layer)
    """
    cumulative_jobs = 0
    
    for layer_idx, info in enumerate(layer_info):
        n_units = info['n_units']
        n_jobs_for_layer = int(np.ceil(n_units / units_per_job))
        
        if job_array_idx < cumulative_jobs + n_jobs_for_layer:
            # This job belongs to this layer
            # Calculate which job within this specific layer
            job_within_layer = job_array_idx - cumulative_jobs
            # Calculate start unit based on position within layer
            start_unit = job_within_layer * units_per_job
            end_unit = min(start_unit + units_per_job, n_units)
            
            return layer_idx, start_unit, end_unit, info['name'], n_units
        
        cumulative_jobs += n_jobs_for_layer
    
    # If we get here, job_array_idx is out of range
    return None, None, None, None, None


def main(args):

    model = args.model_name
    analysis_dir = Path(args.analysis_dir) / model

    h5_fn = analysis_dir / f"{model}_model_activations_0dB_time_avg.h5"

    h5 = h5py.File(h5_fn, 'r')
    print("HDF5 keys:", list(h5.keys()))
    
    # Get all layer names that are actual datasets (not the layer_names metadata)
    # Filter for conv_block layers with relu_target
    all_keys = list(h5.keys())
    layer_names = [key for key in all_keys if isinstance(h5[key], h5py.Dataset) 
                   and 'conv_block' in key and 'relu_target' in key]
    layer_names = sorted(layer_names)  # Ensure consistent ordering
    
    print(f"Found {len(layer_names)} layers to process: {layer_names}")
    
    # Get layer info (number of units per layer)
    layer_info = get_layer_info(h5, layer_names)
    
    # Calculate total jobs needed
    total_jobs = sum(int(np.ceil(info['n_units'] / args.units_per_job)) for info in layer_info)
    print(f"Total jobs needed across all layers: {total_jobs}")
    
    # Determine which layer and units this job should process
    layer_ix, start_unit, end_unit, layer_name, n_units = compute_layer_and_unit_indices(
        args.job_array_idx, layer_info, args.units_per_job
    )
    
    if layer_ix is None:
        print(f"Job {args.job_array_idx}: Out of range (total jobs needed: {total_jobs})")
        h5.close()
        return
    
    # Save results with layer and unit range in filename
    out_path = Path(args.out_dir) / model
    out_path.mkdir(parents=True, exist_ok=True)
    out_name = out_path / f"{layer_name}_jsin_stim_anova_results_units_{start_unit:05d}_{end_unit-1:05d}.npz"

    if out_name.exists() and not args.overwrite:
        print(f"Job {args.job_array_idx}: Output file {out_name} already exists. Skipping computation.")
        h5.close()
        return
        
    print(f"Job {args.job_array_idx}: Processing layer {layer_ix} ({layer_name})")
    print(f"  Units {start_unit} to {end_unit-1} (total {end_unit-start_unit} units out of {n_units})")
    
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


    optimal_bins = optimal_bin_count(target_f0s)
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
    n_examples = layer_acts.shape[0]
    
    # Note: n_units is already computed in compute_layer_and_unit_indices
    # but we verify it matches
    assert n_units == layer_acts.shape[-1], f"Unit count mismatch: expected {n_units}, got {layer_acts.shape[-1]}"
    
    # Calculate unit range for this job
    units_per_job = args.units_per_job
    job_idx = args.job_array_idx
    
    # Check if this job has any units to process
    if start_unit >= n_units:
        print(f"Job {job_idx}: No units to process (start_unit {start_unit} >= n_units {n_units})")
        h5.close()
        return
    
    print(f"Job {job_idx}: Processing units {start_unit} to {end_unit-1} (total {end_unit-start_unit} units)")
    
    # Prepare data for only the units this job will process
    dependent_var = layer_acts.flatten()
    unit_ix = np.arange(n_units)
    unit_ix = np.tile(unit_ix, n_examples)

    act_df = pd.DataFrame(dependent_var, columns=['activation'])
    act_df['unit_ix'] = unit_ix.astype(int)
    act_df['f0'] = np.repeat(f0_list, n_units)
    act_df['location'] = np.repeat(loc_list, n_units)
    act_df['word_int'] = np.repeat(h5['target_word_int'][valid_indices], n_units).astype('int')
    act_df['speaker_int'] = np.repeat(h5['target_talker_id'][valid_indices], n_units).astype('int')
    
    # Initialize arrays for only the units this job processes
    n_units_this_job = end_unit - start_unit
    prop_var_per_unit = np.zeros((n_units_this_job, 4))
    ssq_per_unit = np.zeros((n_units_this_job, 4))
    category_labels = ['f0', 'location', 'word', 'speaker'] # , 'f0:location', 'f0:word', 'location:word']

    # Function to process each unit
    def process_unit(unit_i):
        ssq_per_cat = np.zeros(4)
        prop_var = np.zeros(4)
        for ix, category in enumerate(["C(f0)", "C(location)", "C(word_int)",  "C(speaker_int)"]):
            formula = f"activation ~ {category}"
            model = ols(formula, act_df[act_df.unit_ix == unit_i]).fit()
            anova_table = sm.stats.anova_lm(model)
            total_ss = anova_table.sum_sq.sum()
            ssq = anova_table['sum_sq'][:-1].item()
            ssq_per_cat[ix] = ssq
            prop_var[ix] = (ssq / total_ss)
        return unit_i, prop_var, ssq_per_cat

    # Process only the units assigned to this job
    results = Parallel(n_jobs=args.n_jobs)(
        delayed(process_unit)(unit_i) 
        for unit_i in tqdm(range(start_unit, end_unit), total=n_units_this_job)
    )

    # Collect results (store with local indexing)
    for unit_i, prop_var, ssq in results:
        local_idx = unit_i - start_unit
        prop_var_per_unit[local_idx] = prop_var
        ssq_per_unit[local_idx] = ssq
    

    # save as npz with unit range and layer information

    out_dict = dict(  
        prop_var_per_unit=prop_var_per_unit, 
        ssq_per_unit=ssq_per_unit, 
        category_labels=category_labels,
        layer_name=layer_name,
        layer_ix=layer_ix,
        start_unit=start_unit,
        end_unit=end_unit,
        total_units=n_units
        )
    # save as pickle
    with open(out_name, 'wb') as f:
        pd.to_pickle(out_dict, f)
    
    print(f"Saved results to {out_name}")

    h5.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, help="model name")
    parser.add_argument("--analysis_dir", type=str, help="path to analysis directory")
    parser.add_argument("--out_dir", type=str, help="path to analysis directory")
    parser.add_argument("--job_array_idx", type=int, default=0, 
                        help="SLURM job array index (0-based). This determines both layer and unit range.")
    parser.add_argument("--units_per_job", type=int, default=100, 
                        help="number of units to process per job")
    parser.add_argument("--n_jobs", type=int, default=1, 
                        help="number of parallel workers within each job")
    parser.add_argument("--overwrite", action='store_true',
                        help="whether to overwrite existing output files")
    args = parser.parse_args()
    main(args)