import scipy 
import numpy as np 
import pandas as pd 
from pathlib import Path 
import json
import re

##############################
# For main SWC experiment
##############################
def get_part_df_swc(fname):
    part_data = json.load(open(fname, 'r'))
    # print(f"{fname.stem} success {part_data[0]['success']}")
    part_df = pd.DataFrame.from_records(part_data)
    ## Forward fill stim presentation entry to word response entry
    responses = part_df.loc[part_df.trial_type.isin(['audio-keyboard-response','dictionary-text']), ['trial_index', 'stimulus']]
    responses = responses.ffill()
    part_df.loc[part_df['trial_index'].isin(responses["trial_index"].values), 'stimulus'] = responses.stimulus
    for item in part_data:
        if 'response' in item:
            if  isinstance(item['response'], dict):
                if 'headphones' in item['response']:
                    over_ear = False if item['response']['headphones'] == 'In-ear headphones or earbuds' else True
                if 'hearing_loss' in item['response']:
                    hearing_loss = False if item['response']['hearing_loss'] == "Not aware of any hearing loss" else True
                else:
                    for key, value in item['response'].items():
                        part_df[key] = value 
    part_df['over_ear_hf'] = over_ear
    part_df['hearing_loss'] = hearing_loss
    return part_df

def get_stim_snr_and_cond(stim_str, stim_cond_map=None):
    condition, snr = None,  None 
    if isinstance(stim_str, str) and not stim_str.startswith('<'):
        # print(stim_str)
        cond_str = re.search("condition_(-?\d+)", stim_str)
        if cond_str:
            cond_str = cond_str.group(0)
            condition, snr = stim_cond_map[cond_str]
        elif 'catch' in stim_str:
            condition = 'catch_trial'
            snr = np.inf
    return snr, condition

##############################
# Azim Spotlight fns
##############################
def get_stim_target_azim_and_dist_detla(stim_str, stim_cond_map=None):
    condition, target_azim, dist_delta, distractor_azim = None,  None, None, None 
    if isinstance(stim_str, str) and not stim_str.startswith('<'):
        # print(stim_str)
        cond_str = re.search("condition_(-?\d+)", stim_str)
        if cond_str:
            cond_str = cond_str.group(0)
            condition = 'spatialized'
            target_azim, dist_delta, distractor_azim = stim_cond_map[cond_str]
        elif 'catch' in stim_str:
            condition = 'catch_trial'
        elif 'srm' in stim_str:
            condition = 'srm_trial'
    return condition, target_azim, dist_delta, distractor_azim

###############################
# For speaker array experiment
###############################

def get_target_transcript(fname, df_w_transcripts):
    try:
        return df_w_transcripts.loc[df_w_transcripts['targ_src_stem'].eq(fname), 'target_transcripts'].values[0]
    except IndexError:
        return ['']
        
def get_distractor_tscript(fname, df_w_transcripts):
    tscript = ['']
    if df_w_transcripts['dist_1_src_stem'].eq(fname).any():
        tscript = df_w_transcripts.loc[df_w_transcripts['dist_1_src_stem'].eq(fname), 'distractor_1_transcripts'].values[0]
    elif df_w_transcripts['dist_2_src_stem'].eq(fname).any():
        tscript = df_w_transcripts.loc[df_w_transcripts['dist_2_src_stem'].eq(fname), 'distractor_2_transcripts'].values[0]
    return tscript

def get_array_expmt_sex_cond(row, df_w_transcripts):
    sex_cond = df_w_transcripts.loc[((df_w_transcripts['dist_1_src_stem'].eq(row.distractor_1_fn)) & (df_w_transcripts['dist_2_src_stem'].eq(row.distractor_2_fn)) & (df_w_transcripts['targ_src_stem'].eq(row.src_fn))), 'sex_cond'].values
    if len(sex_cond) > 1:
        print(f"Multiple talker sex conditions: {sex_cond}")
    return sex_cond


# More-generalized functions
##############################

## Get all participant data into one df for analysis
def get_part_df(fname):
    part_df = pd.read_csv(fname)
    part_df = part_df[part_df.trial_type == 'dictionary-text'].reset_index(drop=True)
    part_df.trial_num = part_df.trial_num.astype(float).astype('int')
    # part_df['participant'] = fname.stem
    return part_df

def get_part_df_loc_check(fname):
    part_df = pd.read_csv(fname)
    part_df = part_df[~part_df.trial_num.isna()].reset_index(drop=True)
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
    # azim = abs(azim)
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
    manifest_df['target_azim'] = manifest_df['target_loc'].apply(lambda x: x[1][0] if isinstance(x, list) else x[0])
    manifest_df['target_elev'] = manifest_df['target_loc'].apply(lambda x: x[1][1] if isinstance(x, list) else x[1])
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


