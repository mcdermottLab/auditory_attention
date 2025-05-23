import scipy 
import numpy as np 
from scipy.optimize import curve_fit
from numpy.polynomial import Polynomial
from typing import Optional, Union, Tuple, List, Dict
from scipy import stats
import pandas as pd

import re

################################################
# Psychometric functions written by Mark Saddler 
################################################

def psychometric_function(x, a, mu, sigma):
    """ """
    return a * scipy.stats.norm(mu, sigma).cdf(x)


def psychometric_function_inv(y, a, mu, sigma):
    """ """
    return scipy.stats.norm(mu, sigma).ppf(y / a)


def fit_psychometric_function(x, y, method="trf", p0=None, bounds_from_data=None, **kwargs):
    """ """
    if p0 is None:
        p0 = (1, x[np.argmin(np.abs(np.cumsum(y) / np.sum(y) - 0.5))], 1)
    if bounds_from_data:
        a_bounds = (0, 1) 
        mu_bounds = (x.min(), x.max())
        sigma_bounds = (0, x.max() - x.min())
        lower_bounds = (a_bounds[0], mu_bounds[0], sigma_bounds[0])
        upper_bounds = (a_bounds[1], mu_bounds[1], sigma_bounds[1])
        bounds = (lower_bounds, upper_bounds)
    else:
        bounds = [-np.inf, np.inf]
    try:
        popt, pcov = scipy.optimize.curve_fit(
            lambda _, a, mu, sigma: psychometric_function(_, a, mu, sigma),
            xdata=x,
            ydata=y,
            p0=p0,
            method=method,
            # maxfev=10_000,
            bounds=bounds,
            **kwargs,
        )
    except RuntimeError as e:
        print(e)
        popt = np.ones_like(p0) * np.nan
        pcov = np.ones_like(p0) * np.nan
    return np.squeeze(popt), np.squeeze(pcov)


def compute_srt_from_popt(popt, threshold_value="half"):
    """ """
    if isinstance(threshold_value, str):
        if "half" in threshold_value:
            srt = popt[1]
        else:
            raise ValueError(f"unrecognized {threshold_value=}")
    else:
        srt = psychometric_function_inv(threshold_value, *popt)
    return srt


def estimate_thresholds(x, y, threshold_value="half", **kwargs):
    """ """
    popt, pcov = fit_psychometric_function(x, y, **kwargs)
    srt = compute_srt_from_popt(popt, threshold_value=threshold_value)
    return srt, popt, pcov

##################################################
# Psychometric functions written by Ian Griffith 
##################################################

def fit_threshold_poly(snrs, prop_correct, degree=3):
    poly = Polynomial.fit(x=snrs, y=prop_correct, deg=degree)
    return poly

def get_dBSNR_threshold(poly, threshold=0.5, precision=1_000):
    snrs, prop_correct = poly.linspace(n=precision)
    dB_threshold = snrs[np.argwhere(prop_correct >=threshold).min()]
    return dB_threshold

def estimate_threshold_poly(snrs, prop_correct, degree=2, threshold=0.5, precision=1_000):
    poly = fit_threshold_poly(snrs, prop_correct, degree)
    threshold = get_dBSNR_threshold(poly, threshold, precision)
    return threshold, poly

##################################
# model names for plotting
##################################

def get_model_name(stem):
    str_name = None
    if 'late_only' in stem:
        str_name = 'Late-only'
    elif 'early_only' in stem:
        str_name = 'Early-only'
    elif 'control' in stem:
        str_name = 'Baseline CNN'
    elif 'arch' in stem:
        # get number following pattern 'arch_{n}'
        # e.g. arch_1, arch_2
        arch_n = re.search(r'arch_(\d+)', stem).group(1)
        str_name =  f'Feature-gain alt v{arch_n}'
    elif "main" in stem:
        str_name = 'Feature-gain main'
    elif "50Hz" in stem:
        str_name = '50Hz cutoff'
    elif 'backbone' in stem:
        if 'babble' in stem:
            if 'coloc' in stem:
                str_name = 'Backbone babble all co-located'
            elif 'word_babble_and_noise' in stem:
                str_name = 'Backbone babble + noise'
            else:
                str_name = 'Backbone babble old'
        else:
            str_name = 'Backbone'
        if 'no_gain' in stem:
            str_name = f'{str_name} no gains'
        elif 'learned' in stem:
            str_name = f'{str_name} learned gains'
        elif 'ecdf' in stem and 'feature' not in stem:
            str_name = f"{str_name} computed gains"
        elif 'ecdf' in stem and 'feature' in stem:
            str_name = f'{str_name} computed feature gains'

    if 'rand' in stem:
        str_name = str_name + ' random weights'
    if str_name is None:
        return stem
    return str_name


model_name_dict = {
                   'word_task_v08_control_no_attn': 'Baseline CNN v08',
                   'word_task_v09_control_no_attn': 'Baseline CNN v09',
                   'word_task_early_only_v09': 'Early-only',
                   'word_task_late_only_v09': 'Late-only',
                   "word_task_gender_balanced_fc_1024_v08": "Gender Balanced large fc v08",
                   "word_task_25p_loc_v07_LN_last_valid_time_no_affine": "25% co-located LN last valid time no affine",
                   "word_task_half_co_loc_v08_gender_bal": "Gender Balanced v08",
                   "word_task_half_co_loc_v08_gender_bal_4M": "Gender Balanced v08 4M",
                   "word_task_half_co_loc_v08_gender_bal_4M_orig": "50% co-located v08 4M",
                   "word_task_half_co_loc_v08_gender_bal_4M_sanity": "50% co-located GB v08 4M",
                   "word_task_deep_fc_1024_v08": "Deeper Architecture",
                   "word_task_half_co_locate_deep_fc_1024_v08": "Deep Arch. 50% co-located",
                   "word_task_half_co_locate_deep_fc_1024_v08_old": "Deep Arch. 50% co. old ckpt",
                   "word_task_half_co_loc_v08_gender_bal_4M_w_no_cue_learned": "learned no cue trials",
                   "word_task_half_co_loc_v08_gender_bal_4M_w_no_cue_learned_higher_lr_less_dropout": "Feature-gain model v08",
                   "word_task_half_co_loc_v09_gender_bal_4M_w_no_cue_learned_higher_lr_less_dropout": "Feature-gain model v09",
                   "word_task_conventional_layer_order": "Conventional Layer Order",
                   "word_task_half_co_loc_v09_50Hz_cutoff": "50Hz Cutoff model",
                   "word_task_v09_cue_loc_task": "Dual task model ",
                   "word_task_v10_main_feature_gain_config": "Feature-gain v10 "
                  }

# set order of models for human-model similarity plots 
model_name_order = [
                    'Feature-gain Model',
                    'Baseline CNN',
                    'Early-only',
                    'Late-only',
                ]
model_name_order_w_50Hz = model_name_order + ["50Hz cutoff"]

# set colors for human model similarity
model_color_dict = {

    'Feature-gain Model': '#0FB5AE',
    'Baseline CNN': '#808080',
    'Early-only': '#4046CA',
    'Late-only': '#F68511',
    '50Hz cutoff': '#DE3D82',
}

experiment_color_dict = {

    'Diotic':"tab:pink",
    'Harmonicity':"green",
    'Threshold':"tab:blue",
    'Spotlight':"#9467bd"
}


def annot_stat(star, x1, x2, y, h, col='k', ax=None, lw=1.5):
    ax = plt.gca() if ax is None else ax
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=lw, c=col)
    ax.text((x1+x2)*.5, y+h, star, ha='center', va='bottom', color=col)


def draw_stats_bar(ax, x1, x2, y, h, text, th=0.025, lw=1.5, col='k', fontsize=10, text_gap=0.02):
    # Calculate the position of the text
    text_x = (x1 + x2) * 0.5
    text_y = y + th
    
    # Calculate the gap around the text
    gap_half_width = text_gap * len(text) # * (x2 - x1) * 0.5
    
    # Draw the left part of the bar
    ax.plot([x1, x1, text_x - gap_half_width], [y, y + h, y + h], lw=lw, c=col)
    
    # Draw the right part of the bar
    ax.plot([text_x + gap_half_width, x2, x2], [y + h, y + h, y], lw=lw, c=col)
    
    # Draw the text in the middle of the bar
    ax.text(text_x, y-0.002, text, ha='center', va='center', color=col, fontsize=fontsize)
    

def get_star(p_val):
    if p_val < 0.05:
        text = "*"
    if p_val < 0.01:
        text = "**"
    if p_val < 0.001:
        text = "***"
    if p_val < 0.0001:
        text = "****"
    return text


############################################
# Other util functions by Ian Griffith
############################################

def bootstrap_sem(data, n_bootstraps=1000):
    bootstrapped_means = np.zeros(n_bootstraps)
    n = len(data)
    for i in range(n_bootstraps):
        bootstrapped_sample = np.random.choice(data, size=n, replace=True)
        bootstrapped_means[i] = bootstrapped_sample.mean()
    return bootstrapped_means.std()

# split half reliability
def split_half_reliability(data: pd.DataFrame,
                           groupby_condition: Optional[Union[str, List[str]]] = None,
                           measure_string: Optional[str] = "accuracy",
                           n_splits: Optional[int] = 1000,
                           tqdm: Optional[bool] = False,
                           ) -> Tuple[float, List[float]]:
    """
    Calculate the split-half reliability of a measure.
    The split-half reliability is calculated by splitting the data in half
    and calculating the correlation between the two halves.
    Args:
        data (pd.DataFrame): DataFrame containing the data.
        groupby_condition (str or list of str): Column name(s) to group by.
        measure_string (str): Column name of the measure to calculate reliability for.
        n_splits (int): Number of splits to perform.
        tqdm (bool): Whether to show a progress bar.
    Returns:
        Tuple[float, List[float]]: Split-half reliability and list of reliabilities for each split.
    """

    reliabilities = np.zeros(n_splits)
    if tqdm:
        from tqdm import trange
        iterable = trange(n_splits, desc="Calculating split-half reliability")
    else:
        iterable = range(n_splits)
    for i in iterable:
        split1 = data.sample(frac=0.5)
        split2 = data.drop(split1.index)
        split1 = split1.groupby(groupby_condition)[measure_string].mean().values
        split2 = split2.groupby(groupby_condition)[measure_string].mean().values
        r, p = stats.pearsonr(split1, split2)
        reliabilities[i] = r
    mean_r = np.mean(reliabilities)
    split_half_r = (2*mean_r) / (1 + mean_r)
    return split_half_r, reliabilities