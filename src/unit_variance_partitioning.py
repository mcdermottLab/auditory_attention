import argparse
import numpy as np
import pandas as pd
import h5py
from pathlib import Path
from itertools import combinations
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score
import os
from tqdm import tqdm 

def optimal_bin_count(data, min_bins=1, max_bins=50):
    q25, q75 = np.percentile(data, [25, 75])
    bin_width = 2 * (q75 - q25) * len(data) ** (-1/3)
    bin_width = max(bin_width, 1e-5)
    data_range = np.ptp(data)
    min_bins = max(min_bins, int(np.ceil(data_range / bin_width)))
    max_bins = min(max_bins, int(np.floor(data_range / bin_width)))
    best_bin_count = min_bins
    best_variance = float('inf')
    for bins in range(min_bins, max_bins + 1):
        counts, _ = np.histogram(data, bins=bins)
        variance = np.var(counts)
        if variance < best_variance:
            best_variance = variance
            best_bin_count = bins
    return best_bin_count


def get_layer_info(h5, layer_names):
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
    cumulative_jobs = 0
    for layer_idx, info in enumerate(layer_info):
        n_units = info['n_units']
        n_jobs_for_layer = int(np.ceil(n_units / units_per_job))
        if job_array_idx < cumulative_jobs + n_jobs_for_layer:
            job_within_layer = job_array_idx - cumulative_jobs
            start_unit = job_within_layer * units_per_job
            end_unit = min(start_unit + units_per_job, n_units)
            return layer_idx, start_unit, end_unit, info['name'], n_units
        cumulative_jobs += n_jobs_for_layer
    return None, None, None, None, None


def compute_unique_r2(y, factor_matrix_list):
    """
    Computes true unique R² for each factor using inclusion-exclusion.
    Only returns unique R²s.

    Args:
        y: Target vector (n_examples,)
        factor_matrix_list: list of one-hot encoded design matrices for each factor

    Returns:
        unique_r2s: list of unique R² for each factor
        full_r2: R² of full model
    """
    n_factors = len(factor_matrix_list)
    all_indices = set(range(n_factors))

    # Precompute all R² values for all subsets
    r2_lookup = {}

    for r in range(n_factors + 1):
        for subset in combinations(range(n_factors), r):
            if not subset:
                r2_lookup[subset] = 0.0  # R² of null model
            else:
                X = np.hstack([factor_matrix_list[i] for i in subset])
                model = LinearRegression().fit(X, y)
                r2 = r2_score(y, model.predict(X))
                r2_lookup[subset] = r2

    full_r2 = r2_lookup[tuple(range(n_factors))]

    # Now compute unique R² for each factor
    unique_r2s = []
    for i in range(n_factors):
        rest = tuple(sorted(all_indices - {i}))
        with_i = tuple(sorted(all_indices))
        unique_r2 = r2_lookup[with_i] - r2_lookup[rest]
        unique_r2s.append(unique_r2)

    return full_r2, unique_r2s

# def process_unit(i, y_matrix, factor_matrices):
#     y = y_matrix[:, i]
#     full_r2, unique_r2s = compute_true_unique_r2(y, factor_matrices)
#     return full_r2, unique_r2s

def main(args):
    np.random.seed(0)
    model = args.model_name
    analysis_dir = Path(args.analysis_dir) / model
    h5_fn = analysis_dir / f"{model}_model_activations_0dB_time_avg.h5"
    h5 = h5py.File(h5_fn, 'r', swmr=True)

    all_keys = list(h5.keys())
    layer_names = sorted([key for key in all_keys if isinstance(h5[key], h5py.Dataset)
                          and 'conv_block' in key and 'relu_target' in key])
    print(f"Found {len(layer_names)} layers to process: {layer_names}")

    layer_info = get_layer_info(h5, layer_names)
    total_jobs = sum(int(np.ceil(info['n_units'] / args.units_per_job)) for info in layer_info)
    print(f"Total jobs needed across all layers: {total_jobs}")

    layer_ix, start_unit, end_unit, layer_name, n_units = compute_layer_and_unit_indices(
        args.job_array_idx, layer_info, args.units_per_job
    )

    if layer_ix is None:
        print(f"Job {args.job_array_idx}: Out of range (total jobs needed: {total_jobs})")
        h5.close()
        return

    out_path = Path(args.out_dir) / model
    out_path.mkdir(parents=True, exist_ok=True)
    out_name = out_path / f"{layer_name}_varpart_units_{start_unit:05d}_{end_unit-1:05d}.npz"

    if out_name.exists() and not args.overwrite:
        print(f"Output {out_name} exists. Skipping.")
        h5.close()
        return

    print(f"Job {args.job_array_idx}: Processing layer {layer_ix} ({layer_name})")
    print(f"  Units {start_unit} to {end_unit-1} (total {end_unit - start_unit})")

    ### Load target info and filter valid rows
    target_f0s = h5["target_f0"][:]
    target_locs = h5["target_loc"][:]
    valid_indices = ~np.isnan(target_f0s)

    target_f0s = target_f0s[valid_indices]
    target_locs = target_locs[valid_indices]
    target_words = h5['target_word_int'][valid_indices].astype(int)
    target_speakers = h5['target_talker_id'][valid_indices].astype(int)

    # Bin F0
    optimal_bins = optimal_bin_count(target_f0s)
    counts, bins = np.histogram(target_f0s, bins=optimal_bins)
    f0_assignments = np.digitize(target_f0s, bins, right=True)
    f0_bins = bins[f0_assignments]
    f0_list = np.array([f"{int(b)} Hz" for b in f0_bins])

    # Location labeling
    unique_locations = np.unique(target_locs, axis=0).astype(int)
    location_ixs = {}
    for loc in unique_locations:
        loc_ixs = np.where(np.all(target_locs == loc, axis=1))[0]
        azim, elev = loc
        if elev == 0 and azim == 0:
            label = 'front'
        elif elev == 0:
            label = f"{azim} azim"
        elif azim == 0:
            label = f"{elev} elev"
        else:
            label = f"{azim}_{elev}"
        location_ixs[label] = loc_ixs

    loc_list = np.empty(len(target_locs), dtype=object)
    for label, idxs in location_ixs.items():
        loc_list[idxs] = label

    ### Get layer activations for valid stimuli
    layer_data = h5[layer_name][valid_indices, start_unit:end_unit]
    n_examples = layer_data.shape[0]
    n_units_local = end_unit - start_unit

    results = {
        "unit_indices": np.arange(start_unit, end_unit),
        "total_r2": [],
        "unique_r2s": []
    }

    # Encode factors
    enc = OneHotEncoder(sparse=False, drop='first')
    factor_matrices = [
        enc.fit_transform(f0_list.reshape(-1, 1)),
        enc.fit_transform(loc_list.reshape(-1, 1)),
        enc.fit_transform(target_words.reshape(-1, 1)),
        enc.fit_transform(target_speakers.reshape(-1, 1))
    ]

    # Per-unit variance partitioning
    for i in tqdm(range(n_units_local), desc="Units", unit="unit"):
        y = layer_data[:, i]
        total_r2, unique_r2s = compute_unique_r2(y, factor_matrices)
        results["total_r2"].append(total_r2)
        results["unique_r2s"].append(unique_r2s)

    print(f"Completed variance partitioning for units {start_unit} to {end_unit-1}")
    print(f"  Saving results to {out_name}")
    print(f"Results example (first unit): Total R²={results['total_r2'][0]:.4f}")
    print("  Unique R² factors: F0, Location, Word, Speaker")
    print(f"  Unique R²s={results['unique_r2s'][0]}")

    # Save output
    np.savez_compressed(out_name,
                        unit_indices=results["unit_indices"],
                        total_r2=np.array(results["total_r2"]),
                        unique_r2s=np.array(results["unique_r2s"]))
    print(f"Saved results to {out_name}")
    h5.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Variance partitioning on DNN activations per unit")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--analysis_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--job_array_idx", type=int, required=True)
    parser.add_argument("--units_per_job", type=int, default=50)
    parser.add_argument("--overwrite", action='store_true')
    parser.add_argument("--num_workers", type=int, default=4,
                    help="Number of parallel workers per job (default=4)")
    args = parser.parse_args()
    main(args)
