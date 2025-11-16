import numpy as np 
import pandas as pd
from pathlib import Path 
import re 
from src import util_analysis 
from src import util_process_prolific as util_process
from tqdm.auto import tqdm
import pickle 


def main():
    #############################
    # Set file paths for data
    #############################
    parent_dir = Path("/om/user/imgriff/datasets/human_word_rec_SWC_2024/")
    manifest = pd.read_pickle(parent_dir / "full_cue_target_distractor_df_w_meta_paths.pdpkl")

    ################################
    # Format transcript df 
    ################################
    ### Load in experiment manifests with full transcripts
    path_to_manifest = Path('/om/user/imgriff/datasets/human_word_rec_SWC_2024/full_cue_target_distractor_df_w_meta_transcripts.pdpkl')
    df_w_transcripts = pd.read_pickle(path_to_manifest)
    df_w_transcripts['targ_src_stem'] = df_w_transcripts.excerpt_src_fn.apply(util_process.slice_transcript_path)

    # unpack distractor words to individual columns
    dist_1_word, dist_2_word = zip(*df_w_transcripts.distractor_word.apply(util_process.unpack_dist_words))

    df_w_transcripts['distractor_1_word'] = dist_1_word
    df_w_transcripts['distractor_2_word'] = dist_2_word
    df_w_transcripts['targ_src_stem'] = df_w_transcripts.excerpt_src_fn.apply(util_process.slice_transcript_path)
    df_w_transcripts['dist_1_src_stem'] = df_w_transcripts.excerpt_distractor_1_src_fn.apply(util_process.slice_transcript_path)
    df_w_transcripts['dist_2_src_stem'] = df_w_transcripts.excerpt_distractor_2_src_fn.apply(util_process.slice_transcript_path)

    def get_sex_cond(tgt_name, df_w_transcripts=df_w_transcripts):
        if tgt_name is None:
            return None
        return df_w_transcripts[df_w_transcripts["targ_src_stem"] == tgt_name].sex_cond.item()


    out_dir = Path("final_results_dir")
    path_to_parts = Path('/mindhive/mcdermott/www/imgriff/part_data/binaural_cocktail_party/thresholds_v02')

    part_results = sorted(list(path_to_parts.glob("*.csv")))

    path_to_meta_data = Path('/mindhive/mcdermott/www/imgriff/part_data/binaural_cocktail_party/speaker_array_manifests/thresholds_v02/')
    meta_files = sorted(list(path_to_meta_data.glob("*meta.pkl")))

    #################################
    # Load in the transcripts
    #################################
    # meta_files
    manifest_dict = {}
    for meta_file in meta_files:
        if 'pilot' in meta_file.parent.stem:
            part_name = "pilot_" + "_".join(meta_file.stem.split('_')[:2])

        else:
            part_name = "_".join(meta_file.stem.split('_')[:2])
        manifest_dict[part_name] = meta_file

    #################################
    # Unbox human data
    #################################

    dfs = []
    for result_file in part_results:
        part_name = result_file.stem
        part_ix = int(re.search(r'(\d+)', part_name).group())
        if part_ix > 33:
            continue
        if any(pilot_part_ix == part_ix for pilot_part_ix in [34, 35]):
            continue 

        # remove space 

        if 'pilot' in result_file.parent.stem:
            part_name = "pilot_" + "_".join(result_file.stem.split('_')[:2])
        if ' ' in part_name:
            part_name = part_name.split(' ')[0]
        # get digits in string pattern participant_xxx_ or participant_XXX. 
        part_ix = int(re.search(r'\d+', part_name).group())
        part_name_str = f"participant_{part_ix:03d}"
        manifest_file = manifest_dict[part_name_str]
        part_df = util_process.get_part_df(result_file)
        manifest_df = util_process.get_manifest_df(manifest_file)
        # merge on shared trial_index
        part_df = pd.merge(part_df[['trial_num', 'response', 'correct_response', 'correct']],
                manifest_df, left_on='trial_num', right_on='trial_num', how='left')
        part_df['participant'] = part_name_str
        part_df['tgt_name_stem'] = part_df['src_fn'].apply(lambda x: Path(x).stem)
        part_df['sex_cond'] = part_df['tgt_name_stem'].apply(get_sex_cond)

        dfs.append(part_df)

    results_df = pd.concat(dfs, ignore_index=True)

    ## add transcripts for scoring 
    results_df['dist_1_transcripts'] = [util_process.get_distractor_tscript(fname, df_w_transcripts) for fname in results_df.distractor_1_fn.to_list()]
    results_df['dist_2_transcripts'] = [util_process.get_distractor_tscript(fname, df_w_transcripts) for fname in results_df.distractor_2_fn.to_list()]
    results_df['target_transcripts'] = [util_process.get_target_transcript(fname, df_w_transcripts) for fname in results_df.src_fn.to_list()]

    cols_to_score = ['response', 'target_word', 'distractor_1_word', 'distractor_2_word', 'target_transcripts', 'dist_1_transcripts', 'dist_2_transcripts']

    ## add confusions 
    correct = []
    confusions = []
    for response, target_word, distractor_1_word, distractor_2_word, target_transcripts, dist_1_transcripts, dist_2_transcripts in results_df[cols_to_score].to_numpy():
        correct.append(int(response == target_word or response in target_transcripts))
        confusions.append(int(response in dist_1_transcripts or response in dist_2_transcripts or response in [distractor_1_word, distractor_2_word]))

    results_df['correct'] = correct
    results_df['confusions'] = confusions

    #################################
    # Summarize data per participant
    #################################
    grouped_part_results = results_df.groupby(['participant', 'target_azim', 'target_elev', 'azim_delta', 'sex_cond',
                                   'elev_delta', 'snr', 'n_distractors',]).agg({'correct':['mean', 'sem'],
                                                                                    'confusions':['mean', 'sem', 'count']}).reset_index()
    # flatten multiindex
    grouped_part_results.columns = ['_'.join(col).strip() for col in grouped_part_results.columns.values]
    # remove trailing underscore
    grouped_part_results.columns = [col[:-1] if col[-1] == '_' else col for col in grouped_part_results.columns.values]

    ##################################
    # Filter bad participants 
    ##################################
    part_perf_avg = grouped_part_results.groupby('participant').correct_mean.mean()
    # filter bad participants as those performing below the lower bound (mean - 2 SEM) seen in the online experiments with the same stimuli
    cutoff = 0.3
    good_parts = part_perf_avg[(part_perf_avg > cutoff)].index

    good_part_results = grouped_part_results[grouped_part_results.participant.isin(good_parts)]

    print(f"{good_part_results.participant.nunique()} participants above {cutoff} cutoff (out of {part_perf_avg.shape[0]} total)")

    #################################
    # Fit thresholds via bootstrap
    #################################

    ## Bootstrap over participants to get average and confidence intervals over thresholds
    np.random.seed(0)
    n_boots = 10_000  ## use 10_000 for final analysis

    N = good_part_results.participant.nunique()
    thresholds = []
    for (dist_azim, dist_elev), data in good_part_results.groupby(['azim_delta', 'elev_delta']):
        # break
        print(N, data.participant.nunique())
        for _ in tqdm(range(n_boots)):
            # sample with replacement
            participant_sample = np.random.choice(data.participant.unique(), size=N, replace=True)
            # stack sampled participants - this makes sure data is duplicated if resampled 
            sample_data = pd.concat([data[data.participant == part] for part in participant_sample], axis=0, ignore_index=True).reset_index()
            # average participants per SNR 
            sample_data = sample_data.groupby(['snr']).agg({'correct_mean':'mean'}).reset_index()
            # fit to participant average
            thresh, poly= util_analysis.estimate_threshold_poly(sample_data.snr.values, sample_data.correct_mean.values, degree=2)
            thresholds.append({
                        # 'target_elev': target_elev,
                        'azim_delta':dist_azim, 'elev_delta':dist_elev,
                        'threshold':thresh})
            
    human_thresh_df = pd.DataFrame(thresholds)
    N = good_part_results.participant.nunique()
    human_thresh_df['n_participants'] = N
    # save as df 
    human_thresh_df.to_pickle(out_dir / "summary_2024_human_threshold_results_avg_sex_cond.pdpkl")
    print("Pooled human thresholds calculated and saved")

    ### Bootstrap for sex conditions 
    np.random.seed(0)
    N = good_part_results.participant.nunique()
    thresholds = []
    for (dist_azim, dist_elev, sex_cond), data in good_part_results.groupby(['azim_delta', 'elev_delta', 'sex_cond']):
        # break
        print(N, data.participant.nunique())
        for _ in tqdm(range(n_boots)):
            # sample with replacement
            participant_sample = np.random.choice(data.participant.unique(), size=N, replace=True)
            # stack sampled participants - this makes sure data is duplicated if resampled 
            sample_data = pd.concat([data[data.participant == part] for part in participant_sample], axis=0, ignore_index=True).reset_index()
            # average participants per SNR 
            sample_data = sample_data.groupby(['snr']).agg({'correct_mean':'mean'}).reset_index()
            # fit to participant average
            thresh, poly= util_analysis.estimate_threshold_poly(sample_data.snr.values, sample_data.correct_mean.values, degree=2)
            thresholds.append({
                        # 'target_elev': target_elev,
                        'azim_delta':dist_azim, 'elev_delta':dist_elev,
                        'sex_cond':sex_cond,
                        'threshold':thresh})

    human_thresh_df_sex_conds = pd.DataFrame(thresholds)
    human_thresh_df_sex_conds['n_participants'] = N
    human_thresh_df_sex_conds.to_pickle(out_dir / "summary_2024_human_threshold_results_split_by_sex_cond.pdpkl")
    print("Per-sex human thresholds calculated and saved")

    print("Human thresholds calculated and saved")

    ##########################################
    # Gen null distribution for diff of diffs 
    ##########################################
    ## Bootstrap over participants to get average and confidence intervals over thresholds
    np.random.seed(0)
    n_boots = 10_000  

    diff_of_diff_10_dist = np.zeros(n_boots)
    diff_of_diff_60_dist =  np.zeros(n_boots)

    for ix in tqdm(range(n_boots)):
        # permute distractor elevations and azimuths
        permuted = good_part_results.copy()
        # for delta in [10, 60]:
        for part in permuted.participant.unique():
            permuted.loc[permuted.participant == part, 'elev_delta'] = np.random.permutation(permuted[permuted.participant == part].elev_delta)
            permuted.loc[permuted.participant == part, 'azim_delta'] = np.random.permutation(permuted[permuted.participant == part].azim_delta)
        
        # average over particiapnts 
        permuted_summary = permuted.groupby(['snr', 'elev_delta', 'azim_delta' ]).agg({'correct_mean':'mean'}).reset_index()
        
        # only need to do this once per permutation
        zero_delta = permuted_summary[(permuted_summary.elev_delta == 0) & (permuted_summary.azim_delta == 0)]
        thresh_0, _ = util_analysis.estimate_threshold_poly(zero_delta.snr.values, zero_delta.correct_mean.values, degree=2)

        # get elev delta threshold 
        elev_delta = permuted_summary[(permuted_summary.elev_delta == 10)]
        thresh_elev, _ = util_analysis.estimate_threshold_poly(elev_delta.snr.values, elev_delta.correct_mean.values, degree=2)
        # get azim delta threshold
        azim_delta = permuted_summary[permuted_summary.azim_delta == 10]
        thresh_azim, _ = util_analysis.estimate_threshold_poly(azim_delta.snr.values, azim_delta.correct_mean.values, degree=2)
        # compute difference
        elev_diff = thresh_elev - thresh_0
        azim_diff = thresh_azim - thresh_0
        # compute difference of differences
        diff_of_diff = azim_diff - elev_diff
        diff_of_diff_10_dist[ix] = diff_of_diff

        #  repeat for 60 degree delta
        elev_delta = permuted_summary[(permuted_summary.elev_delta == 60)]
        thresh_elev, _ = util_analysis.estimate_threshold_poly(elev_delta.snr.values, elev_delta.correct_mean.values, degree=2)
        # get azim delta threshold
        azim_delta = permuted_summary[permuted_summary.azim_delta == 60]
        thresh_azim, _ = util_analysis.estimate_threshold_poly(azim_delta.snr.values, azim_delta.correct_mean.values, degree=2)
        # compute difference
        elev_diff = thresh_elev - thresh_0
        azim_diff = thresh_azim - thresh_0
        # compute difference of differences
        diff_of_diff = azim_diff - elev_diff
        diff_of_diff_60_dist[ix] = diff_of_diff

        null_dist_dict = {'diff_of_diff_10_dist':diff_of_diff_10_dist, 'diff_of_diff_60_dist':diff_of_diff_60_dist}
        with open(out_dir / "threshold_diff_of_diff_null_distributions.pkl", 'wb') as f:
            pickle.dump(null_dist_dict, f)


if __name__ == "__main__":
    main()