import numpy as np 
import h5py
from tqdm.auto import tqdm
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from pathlib import Path
import src.util_analysis as util_analysis
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='')
parser.add_argument('--job_ix', type=int, default=0)
parser.add_argument('--time_avg', action='store_true')
args = parser.parse_args()

vec_pearsonr = np.vectorize(pearsonr,
                signature='(n),(n)->(),()')

vec_spearmanr = np.vectorize(spearmanr,
                signature='(n),(n)->(),()')

def per_row_cosine_similarity(matrix_a, matrix_b):
    """
    Calculates the per-row cosine similarity between two matrices.

    Args:
        matrix_a (numpy.ndarray): The first matrix.
        matrix_b (numpy.ndarray): The second matrix.

    Returns:
        numpy.ndarray: An array containing the cosine similarity for each row.
    """
    if matrix_a.shape[0] != matrix_b.shape[0]:
        raise ValueError("Matrices must have the same number of rows")
    
    similarity_scores = np.array([cosine_similarity(matrix_a[i].reshape(1, -1), matrix_b[i].reshape(1, -1))[0, 0] for i in range(matrix_a.shape[0])])
    return similarity_scores


def get_model_name(stem):
    if 'late_only' in stem:
        return 'Late-only'
    elif 'early_only' in stem:
        return 'Early-only'
    elif 'control' in stem:
        return 'Baseline CNN'
    elif 'arch' in stem:
        arch_n = stem.split('_')[-1]
        return f'Feature-gain alt v{arch_n}'
    elif "main" in stem:
        return 'Feature-gain main'


# for path in paths:
distractor_conditions = ['same_sex_talker', 'diff_sex_talker', 'natural_scene']

model_condition_dict = {0: 'word_task_v10_main_feature_gain_config_latest_ckpt',
                        1: 'word_task_early_only_v10',
                        2: 'word_task_late_only_v10',
                        3: 'word_task_v10_4MGB_ln_first_arch_1',
                        4: 'word_task_v10_4MGB_ln_first_arch_10',
                        5: 'word_task_v10_4MGB_ln_first_arch_12',
                        6: 'word_task_v10_4MGB_ln_first_arch_2',
                        7: 'word_task_v10_4MGB_ln_first_arch_4',
                        8: 'word_task_v10_4MGB_ln_first_arch_6',
                        9: 'word_task_v10_4MGB_ln_first_arch_7',
                        10: 'word_task_v10_4MGB_ln_first_arch_8',
                        11: 'word_task_v10_4MGB_ln_first_arch_9',
                        12: 'word_task_v10_control_no_attn',
                        13: 'word_task_v10_main_feature_gain_config_latest_ckpt_rand_weights'}


h5_dir = Path(f"/om/scratch/Thu/imgriff/binaural_unit_activation_analysis")
result_dir = Path("binaural_unit_activation_analysis")
if args.model == '':
    model = model_condition_dict[args.job_ix]
else:
    model = args.model # "word_task_v10_main_feature_gain_config"

print(f"Running {model}")

if args.time_avg:
    h5_path = h5_dir / f"{model}/{model}_model_activations_0dB_time_avg_diotic.h5"
else:
    h5_path = h5_dir / f"{model}/{model}_model_activations_0dB_diotic.h5"

print("Reading ", h5_path)
with h5py.File(h5_path, 'r') as acts:
    all_act_keys = list(acts.keys())
    ## Get keys with corr in them 
    layer_names = set([l.split('_target')[0] for l in all_act_keys if 'target' in l and not  any([str_part in l for str_part in ['_f0', '_loc', '_word', '_cued']])]) # set to remove duplicates 
    print(layer_names)
    dfs = []
    for layer in tqdm(layer_names):
        ## Get good unit ixs first 
        unit_acts = acts[f"{layer}_target"][:].sum(0)
        good_units = np.where(unit_acts > 0)[0]
        fg_key = f"{layer}_target"
        N_examples = acts[fg_key].shape[0]

        fg_acts = acts[fg_key][:][:,good_units]

        for dist_cond in distractor_conditions:
            # set keys based on condition 
            if dist_cond == 'same_sex_talker':
                bg_key = f"{layer}_cued_same_sex_dist"
                mix_key = f"{layer}_mixture_same"
                fg_corr_key = f"{layer}_cued_target_mixture_same_corr"
                bg_corr_key = f"{layer}_cued_same_sex_dist_mixture_same_corr"

            elif dist_cond == 'diff_sex_talker':
                bg_key = f"{layer}_cued_diff_sex_dist"
                mix_key = f"{layer}_mixture_diff"
                fg_corr_key = f"{layer}_cued_target_mixture_diff_corr"
                bg_corr_key = f"{layer}_cued_diff_sex_dist_mixture_diff_corr"

            elif dist_cond == 'natural_scene':
                bg_key = f"{layer}_cued_nat_scene_dist"  
                mix_key = f"{layer}_mixture_nat_scene" 
                fg_corr_key = f"{layer}_cued_target_mixture_nat_scene_corr"
                bg_corr_key = f"{layer}_cued_nat_scene_dist_mixture_nat_scene_corr"

            # filter key name for cochleagram
            if layer == 'cochleagram':
                bg_corr_key = bg_corr_key.replace('sex_', '') 
                fg_corr_key = fg_corr_key.replace('cued_', '')
                bg_corr_key = bg_corr_key.replace('cued_', '')
                bg_key = bg_key.replace('cued_', '')
                mix_key = mix_key.replace('cued_', '')
            # load activations 
            bg_acts = acts[bg_key][:][:, good_units]
            mix_acts = acts[mix_key][:][:, good_units]

            # get existing corrs 
            fg_corrs_full = acts[fg_corr_key][:, 0]
            bg_corrs_full = acts[bg_corr_key][:, 0]

            # get new corrs 
            fg_corr, _ = vec_pearsonr(fg_acts, mix_acts)
            bg_corr, _ = vec_pearsonr(bg_acts, mix_acts)

            # # get spearman's corr
            # fg_rho, _ = vec_spearmanr(fg_acts, mix_acts)
            # bg_rho, _ = vec_spearmanr(bg_acts, mix_acts)

            # get cosine similarity
            fg_cos = per_row_cosine_similarity(fg_acts, mix_acts)
            bg_cos = per_row_cosine_similarity(bg_acts, mix_acts)

            data_dict = {}
            # add raw/full corr
            data_dict['fg_corrs_full'] = fg_corrs_full
            data_dict['bg_corrs_full'] = bg_corrs_full

            # add filtered corrs
            data_dict['fg_corrs_filt_du'] = fg_corr
            data_dict['bg_corrs_filt_du'] = bg_corr

            # # add filtered rho
            # data_dict['fg_rhos_filt_du'] = fg_rho
            # data_dict['bg_rhos_filt_du'] = bg_rho

            # add filtered cosine similarity
            data_dict['fg_cos_filt_du'] = fg_cos
            data_dict['bg_cos_filt_du'] = bg_cos
            
            data_dict['layer'] = [layer] * N_examples
            data_dict['distractor_condition'] = [dist_cond] * N_examples
            df = pd.DataFrame(data_dict)
            df['model_name'] = get_model_name(h5_path.parent.name)
            dfs.append(df)

act_results  = pd.concat(dfs, ignore_index=True)
act_results['layer'] = act_results['layer'].str.replace('_block_', '')

# act_results['raw_log_corr_ratio'] = np.log(np.sqrt(act_results['fg_corrs_full']**2 / act_results['fg_corrs_full']**2))
# act_results['filt_log_corr_ratio'] = np.log(np.sqrt(act_results['fg_corrs_filt_du']**2 / act_results['bg_corrs_filt_du']**2))

act_corrs = pd.melt(act_results,
                    id_vars = ['layer', 'distractor_condition', 'model_name'],
                    value_vars = ['fg_corrs_filt_du', 'bg_corrs_filt_du', "fg_corrs_full", "bg_corrs_full", 'fg_cos_filt_du', 'bg_cos_filt_du'], # 'fg_rhos_filt_du', 'bg_rhos_filt_du', 
                    var_name='Stream',
                    value_name="sim_measure")


act_corrs['Stream'].replace({'fg_corrs_filt_du': 'corr(target$_i$, mixture$_i$)',
                             'bg_corrs_filt_du': 'corr(distractor$_i$, mixture$_i$)',
                             'fg_cos_filt_du': 'sim(target$_i$, mixture$_i$)',
                             'bg_cos_filt_du': 'sim(distractor$_i$, mixture$_i$)',
                             'fg_corrs_full': 'corr(target$_i$, mixture$_i$) raw',
                             'bg_corrs_full': 'corr(distractor$_i$, mixture$_i$) raw',
                            #  'fg_rhos_filt_du': 'rho(target$_i$, mixture$_i$)',
                            #  'bg_rhos_filt_du': 'rho(distractor$_i$, mixture$_i$)'},
                            },
                             inplace=True)

# act_corrs['Stream'].replace('fg_corrs_filt_du','corr(target$_i$, mixture$_i$)',inplace=True)
# act_corrs['Stream'].replace('bg_corrs_filt_du','corr(distractor$_i$, mixture$_i$)',inplace=True)
# act_corrs['Stream'].replace('fg_corrs_full','corr(target$_i$, mixture$_i$) raw',inplace=True)
# act_corrs['Stream'].replace('bg_corrs_full','corr(distractor$_i$, mixture$_i$) raw',inplace=True)


act_corrs['distractor_condition'].replace('same_sex_talker','Same sex',inplace=True)
act_corrs['distractor_condition'].replace('diff_sex_talker','Different sex',inplace=True)
act_corrs['distractor_condition'].replace('natural_scene','Natural scene',inplace=True)

if args.time_avg:
    out_name = result_dir / f"{model}/{model}_corrs_skip_dead_units_time_avg.pdpkl"
else:
    out_name = result_dir / f"{model}/{model}_corrs_skip_dead_units_full_rep.pdpkl"
out_name.parent.mkdir(parents=True, exist_ok=True)
act_corrs.to_pickle(out_name)