import pandas as pd
import numpy as np 
import multiprocessing as mp
import scipy.stats as stats
from tqdm.auto import tqdm
import sys
import util_analysis
from pathlib import Path 
import re

## Get all participant data into one df for analysis
def get_part_df(fname):
    part_df = pd.read_csv(fname)
    part_df = part_df[part_df.trial_type == 'dictionary-text'].reset_index(drop=True)
    part_df.trial_num = part_df.trial_num.astype(float).astype('int')
    # part_df['participant'] = fname.stem
    return part_df

def slice_transcript_path(path_str):
    return Path(path_str).stem

def unpack_loc_tuple(loc_tup):
    if isinstance(loc_tup[0], tuple):
        n_dist = 2 
        loc_tup = loc_tup[0]
    else:
        n_dist = 1
    azim, elev = loc_tup 
    azim = abs(azim)
    return azim, elev, n_dist

def unpack_dist_words(dist_word_list):
    if isinstance(dist_word_list, list):
        dist_1_word, dist_2_word = dist_word_list
    else:
        dist_1_word, dist_2_word = dist_word_list, None
    return dist_1_word, dist_2_word

def unpack_trial_from_array_manifest(trial):
    target_loc = trial[0]
    distractor_loc = trial[1]
    snr = trial[2]
    cue_src_fn = trial[3][0].stem
    src_fn = trial[4][0].stem
    dist_fn = trial[5]
    
    if len(dist_fn) == 1:
        distractor_1_fn = dist_fn[0].stem
        distractor_2_fn = None
    else:
        distractor_1_fn = dist_fn[0].stem
        distractor_2_fn =dist_fn[1].stem

    return target_loc, distractor_loc, snr, cue_src_fn, src_fn, distractor_1_fn, distractor_2_fn

def get_manifest_df(fname):
    manifest = pd.read_pickle(fname)
    array_manifest_fname = str(fname).replace("meta", "array_manifest")
    trial_manifest = pd.read_pickle(array_manifest_fname)
    # add trial_manifest contents to manifest 
    _, _, _, _, src_fn, dist_1_fn, dist_2_fn = zip(*[unpack_trial_from_array_manifest(trial) for trial in trial_manifest])
    manifest_df = pd.concat([pd.DataFrame(val.values()) for val in manifest.values()]).reset_index(drop=True)
    trial_nums =  manifest_df.index.to_list()
    manifest_df['trial_num'] = trial_nums
    # unpack locations to azimuth and elevation 
    manifest_df['target_azim'] = manifest_df['target_loc'].apply(lambda x: x[0])
    manifest_df['target_elev'] = manifest_df['target_loc'].apply(lambda x: x[1])
    dist_azim, dist_elev, n_dist = zip(*manifest_df['distractor_loc'].apply(lambda x: unpack_loc_tuple(x)))
    manifest_df['distractor_azim'] = dist_azim
    manifest_df['distractor_elev'] = dist_elev
    manifest_df['n_distractors'] = n_dist
    # unpack distractor words 
    dist_1_word, dist_2_word = zip(*manifest_df['distractor_word'].apply(unpack_dist_words))
    manifest_df['distractor_1_word'] = dist_1_word
    manifest_df['distractor_2_word'] = dist_2_word
    # add source and distractor file names
    manifest_df['src_fn'] = [src_fn[ix] for ix in trial_nums]
    manifest_df['distractor_1_fn'] = [dist_1_fn[ix] for ix in trial_nums]
    manifest_df['distractor_2_fn'] = [dist_2_fn[ix] for ix in trial_nums]
    return manifest_df

def get_info_from_trial_dict(fname):
    manifest = pd.read_pickle(fname)
    trials = []
    for ix, trial in enumerate(manifest):
        trial_dict = {}
        trial_dict['target_azim'] = trial[0][0]
        trial_dict['target_elev'] = trial[0][1]
        trial_dict['distractor_azim'] = trial[1][0]
        trial_dict['distractor_elev'] = trial[1][1]
        trial_dict['target_word'] = Path(trial[3]).stem.split('_')[0]
        trial_dict['distractor_word'] = Path(trial[4]).stem.split('_')[0]
        trial_dict['trial_num'] = ix
        trials.append(trial_dict)
    manifest_df = pd.DataFrame(trials)
    return manifest_df

def sample_thresholds(args):
    group_names, k, group_data = args
    ci_group_names = np.random.choice(group_names, k, replace=True)
    ci_data = group_data[group_data.participant.isin(ci_group_names)].groupby('snr').agg({'correct_mean':'mean'}).reset_index()
    try:
        ci_thresh, _ = util_analysis.estimate_threshold_poly(ci_data.snr.values, ci_data.correct_mean.values)
    except:
        ci_thresh = np.nan
    return ci_thresh

def bootstrap_ci(k, participants, n_ci_boots, azim_data):
    group_names = np.random.choice(participants, k, replace=True)
    group_data = azim_data[azim_data.participant.isin(group_names)]
    with mp.Pool(mp.cpu_count()) as pool:
        ci_threshs = np.array(pool.map(sample_thresholds, [(group_names, k, group_data) for _ in range(n_ci_boots)]))
    lower_ci, upper_ci = np.nanpercentile(ci_threshs, [2.5,97.5])
    mean_thresh = np.nanmean(ci_threshs)
    snr_resolvability = upper_ci - lower_ci
    return mean_thresh, snr_resolvability, upper_ci, lower_ci

def get_sample_size_from_job_index(k):
    """ Gets the sample size corresponding to the input job index k.
    Subests are in ranges of 100 jobs, with the first 10 jobs corresponding to a sample size of 2, the next 10 jobs corresponding to a sample size of 3, and so on.
    """
    if k < 100:
        return 3
    elif k >= 100 and k < 200:
        return 4
    elif k >= 200 and k < 300:
        return 5
    elif k >= 300 and k < 400:
        return 6
    elif k >= 400 and k < 500:
        return 7
    elif k >= 500 and k < 600:
        return 8
    elif k >= 600 and k < 700:
        return 9
    else:
        raise ValueError("k must be less than 600.")

def main(k):
    # Set random seed
    np.random.seed(k)
    # get sample size
    samp_size = get_sample_size_from_job_index(k)
    subset = k % 100

    print(f"Running job {k} with sample size {samp_size} and subset {subset}.")
    # Load the data
    parent_dir = Path("/om/user/imgriff/datasets/human_word_rec_SWC_2024/")
    # manifest = pd.read_pickle(parent_dir / "full_cue_target_distractor_df_w_meta_paths.pdpkl")

    outdir = Path('human_loc_power_analysis_v2/')
    outdir.mkdir(exist_ok=True, parents=True)
    out_name = outdir / f'conf_interval_sizes_sample_size_{samp_size}_subset_{subset}.npy'

    if out_name.exists():
        print(f"Output file {out_name} already exists. Exiting.")
        return 
    # get participant results 
    path_to_parts = Path('/mindhive/mcdermott/www/imgriff/part_data/binaural_cocktail_party/thresholds_v01')
    part_results = sorted(list(path_to_parts.glob("*.csv")))

    path_to_meta_data = Path('/mindhive/mcdermott/www/imgriff/part_data/binaural_cocktail_party/speaker_array_manifests/thresholds_v01/')
    meta_files = sorted(list(path_to_meta_data.glob("*meta.pkl")))

    ## add pilot results 
    path_to_parts = Path('/mindhive/mcdermott/www/imgriff/part_data/binaural_cocktail_party/pilot_thresholds_v00')
    part_results += sorted(list(path_to_parts.glob("*.csv")))

    path_to_meta_data = Path('/mindhive/mcdermott/www/imgriff/part_data/binaural_cocktail_party/speaker_array_manifests/pilot_thresholds_v00/')
    meta_files += sorted(list(path_to_meta_data.glob("*meta.pkl")))

    # meta_files
    manifest_dict = {}
    for meta_file in meta_files:
        if 'pilot' in meta_file.parent.stem:
            part_name = "pilot_" + "_".join(meta_file.stem.split('_')[:2])

        else:
            part_name = "_".join(meta_file.stem.split('_')[:2])
        manifest_dict[part_name] = meta_file

    dfs = []
    for result_file in part_results:
        part_name = result_file.stem
        # remove space 
        if 'pilot' in result_file.parent.stem:
            part_name_str = "pilot_" + "_".join(result_file.stem.split('_')[:2])
        else:
        # get digits in string pattern participant_xxx_ or participant_XXX. 
            part_ix = int(re.search(r'\d+', part_name).group())
            part_name_str = f"participant_{part_ix:03d}"
        if ' ' in part_name:
            part_name_str = part_name.split(' ')[0]

        manifest_file = manifest_dict[part_name_str]
        part_df = get_part_df(result_file)
        manifest_df = get_manifest_df(manifest_file)
        # merge on shared trial_index
        part_df = pd.merge(part_df[['trial_num', 'response', 'correct_response', 'correct']],
                manifest_df, left_on='trial_num', right_on='trial_num', how='left')
        part_df['participant'] = part_name_str
        dfs.append(part_df)

    results_df = pd.concat(dfs, ignore_index=True)

    ## add confusions 
    confusions = []
    for response, distractor_word in results_df[['response', 'distractor_word']].to_numpy():
        if isinstance(distractor_word, list):
            confusions.append(int(response in distractor_word))
        else:
            confusions.append(int(response == distractor_word ))

    results_df['confusions'] = confusions
    ## clean up pilot trial
    results_df = results_df[~((results_df.participant == 'pilot_participant_002') & (results_df.trial_num == 359) & (results_df.response == 'trees'))].reset_index(drop=True)
    ## Remove new participants
    pilot_participants = ['participant_001', 'participant_002', 'participant_003', 'participant_004',
                         'participant_005', 'participant_034', 'participant_035', 'pilot_participant_001', 'pilot_participant_002']
    
    results_df = results_df[results_df.participant.isin(pilot_participants)]

    results_df['distractor_elev_delta'] = (results_df['distractor_elev'] - results_df['target_elev']).abs()

    grouped_part_results = results_df.groupby(['participant', 'target_azim', 'target_elev', 'distractor_azim',
                                   'distractor_elev_delta', 'snr', 'n_distractors',]).agg({'correct':['mean', 'sem'],
                                                                                    'confusions':['mean', 'sem', 'count']}).reset_index()
    # flatten multiindex
    grouped_part_results.columns = ['_'.join(col).strip() for col in grouped_part_results.columns.values]
    # remove trailing underscore
    grouped_part_results.columns = [col[:-1] if col[-1] == '_' else col for col in grouped_part_results.columns.values]


    # Load and preprocess the data
    azim_data =  grouped_part_results[(grouped_part_results.distractor_elev_delta == 0) & (grouped_part_results.distractor_azim==0)].copy()
    azim_data = azim_data[(azim_data.distractor_elev_delta == 0) & (azim_data.distractor_azim==0)]
    azim_data = azim_data[(azim_data.n_distractors == 2)]
    azim_data = azim_data[azim_data.snr.isin(np.arange(-9, 7))]

    n_parts = azim_data.participant.nunique()
    unique_participants = azim_data.participant.unique()

    n_sample_size_boots = 100
    n_ci_boots = 10_000

    # will be 2d array with shape (n_sample_size_boots, (mean, CI size, upper ci, lowe ci))
    conf_interval_sizes = np.array([bootstrap_ci(samp_size, unique_participants, n_ci_boots, azim_data) for _ in tqdm(range(n_sample_size_boots))])

    # Save the results

    np.save(out_name, conf_interval_sizes)

if __name__ == '__main__':
    k = int(sys.argv[1])
    main(k)