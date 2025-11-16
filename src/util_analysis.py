import scipy 
import numpy as np 
from scipy.optimize import curve_fit
from numpy.polynomial import Polynomial
from typing import Optional, Union, Tuple, List, Dict
from scipy import stats
import pandas as pd
import seaborn as sns 
import re
import scipy.stats as stats
from tqdm import tqdm
import warnings
from pingouin import rm_anova


MARKER_SIZE = 8

def sign_test(x, mu0):
    n = len(x)
    n_pos = np.sum(x > mu0)
    n_neg = np.sum(x < mu0)
    effect_m = (n_pos - n_neg) / 2 
    p = stats.binomtest(min(n_pos, n_neg), n, p=0.5).pvalue
    return effect_m, p, n_pos, n_neg


# function to calculate Cohen's d for independent samples
def cohend(d1, d2):
	# calculate the size of samples
	n1, n2 = len(d1), len(d2)
	# calculate the variance of the samples
	s1, s2 = np.var(d1, ddof=1), np.var(d2, ddof=1)
	# calculate the pooled standard deviation
	s = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
	# calculate the means of the samples
	u1, u2 = np.mean(d1), np.mean(d2)
	# calculate the effect size
	return (u1 - u2) / s

## Cohen's d for paired samples
def cohend_paired(d1, d2):
    # is difference of means, divided by standard deviation of differences
    return (np.mean(d1) - np.mean(d2))/ np.std(d1 - d2, ddof=1)


def bootstrap_partial_eta_ci(
    df: pd.DataFrame,
    dv_col: str,
    subject_col: str,
    within_factors: list[str],
    n_bootstrap: int = 2000,
    ci_level: float = 0.95,
    random_state: int | None = None,
    tqdm_desc: str = "Partial η² bootstrap",
    suppress_future_warnings: bool = True,
):
    """Run pingouin rm_anova and add bootstrap CIs for partial eta squared."""
    if suppress_future_warnings:
        warnings.filterwarnings("ignore", category=FutureWarning)

    rng = np.random.default_rng(random_state)

    # Base ANOVA
    anova_table = rm_anova(
        data=df,
        dv=dv_col,
        subject=subject_col,
        within=within_factors,
        effsize="np2",
    )

    subjects = df[subject_col].unique()
    bootstrap_np2 = {src: [] for src in anova_table["Source"]}

    for _ in tqdm(range(n_bootstrap), desc=tqdm_desc):
        sampled_ids = rng.choice(subjects, size=len(subjects), replace=True)
        boot_df = df[df[subject_col].isin(sampled_ids)].copy()

        try:
            boot_table = rm_anova(
                data=boot_df,
                dv=dv_col,
                subject=subject_col,
                within=within_factors,
                effsize="np2",
            )
            for src, np2 in zip(boot_table["Source"], boot_table["np2"]):
                bootstrap_np2[src].append(np2)
        except Exception:
            continue  # skip occasional singular fits

    alpha = 1 - ci_level
    lower = alpha / 2 * 100
    upper = (1 - alpha / 2) * 100

    anova_table["np2_CI_lower"] = [
        np.percentile(bootstrap_np2[src], lower) if bootstrap_np2[src] else np.nan
        for src in anova_table["Source"]
    ]
    anova_table["np2_CI_upper"] = [
        np.percentile(bootstrap_np2[src], upper) if bootstrap_np2[src] else np.nan
        for src in anova_table["Source"]
    ]

    return anova_table, bootstrap_np2


def bootstrap_paired_ttest_cohens_d(
    x: np.ndarray,
    y: np.ndarray,
    n_bootstrap: int = 2000,
    ci_level: float = 0.95,
    random_state: int | None = None,
):
    """Return paired t-test stats, Cohen's d, and bootstrap CI for d."""
    assert len(x) == len(y), "paired samples must have same length"
    rng = np.random.default_rng(random_state)

    # point estimates
    t_res = stats.ttest_rel(x, y, nan_policy="raise")
    diff = x - y
    cohens_d = diff.mean() / diff.std(ddof=1)

    # bootstrap
    boot_ds = []
    idx = np.arange(len(x))
    for _ in range(n_bootstrap):
        sample_idx = rng.choice(idx, size=len(idx), replace=True)
        diff_boot = (x[sample_idx] - y[sample_idx])
        sd_boot = diff_boot.std(ddof=1)
        if sd_boot == 0:  # guard against zero variance
            continue
        boot_ds.append(diff_boot.mean() / sd_boot)

    alpha = 1 - ci_level
    lower = np.percentile(boot_ds, alpha / 2 * 100)
    upper = np.percentile(boot_ds, (1 - alpha / 2) * 100)

    return {
        "t": t_res.statistic,
        "df": t_res.df,
        "p": t_res.pvalue,
        "d": cohens_d,
        "d_ci_lower": lower,
        "d_ci_upper": upper,
        "n_bootstrap": len(boot_ds),
    }


def bootstrap_sign_test_summary(
    baseline_values: np.ndarray,
    comparison_values: dict[str, np.ndarray],
    n_bootstrap: int = 2000,
    random_state: int | None = None,
):
    """
    Bootstrap the sign-test pipeline used for r² / RMSE comparisons.

    Parameters
    ----------
    baseline_values : np.ndarray
        Distribution you currently compare every model against (e.g., pooled
        feature-gain values).
    comparison_values : dict[str, np.ndarray]
        Mapping {model_name: value_array}. Each array plays the role of `mu0`
        in the original sign test.
    n_bootstrap : int
        Number of bootstrap resamples at the participant/model level.
    random_state : int | None
        Seed for reproducibility.

    Returns
    -------
    dict
        {
          model_name: {
            "diff_mean": …,
            "n_pos": …,
            "n_neg": …,
            "n_total": …,
            "sign_test_stat": …,
            "sign_test_p": …,
            "boot_ci_low": …,
            "boot_ci_high": …,
            "boot_samples": np.ndarray([...])
          },
          …
        }
    """
    rng = np.random.default_rng(random_state)
    results = {}

    for model, model_vals in comparison_values.items():
        # point estimate (matches existing cell)
        stat, p, n_pos, n_neg = sign_test(baseline_values, mu0=model_vals)
        diff = baseline_values.mean() - model_vals.mean()

        # bootstrap resampling across participants/models
        boot_diffs = []
        boot_stats = []
        boot_ps = []

        n_baseline = len(baseline_values)
        n_model = len(model_vals)

        for _ in range(n_bootstrap):
            b_idx = rng.choice(n_baseline, n_baseline, replace=True)
            m_idx = rng.choice(n_model, n_model, replace=True)

            boot_baseline = baseline_values[b_idx]
            boot_model = model_vals[m_idx]

            boot_diffs.append(boot_baseline.mean() - boot_model.mean())
            st, pv, n_pos, n_neg = sign_test(boot_baseline, mu0=boot_model)
            boot_stats.append(st)
            boot_ps.append(pv)

        ci_low = np.percentile(boot_diffs, 2.5)
        ci_high = np.percentile(boot_diffs, 97.5)

        results[model] = {
            "diff_mean": diff,
            "sign_test_stat": stat,
            "n_pos": n_pos,
            "n_neg": n_neg,
            "n_total": n_baseline,
            "sign_test_p": p,
            "boot_ci_low": ci_low,
            "boot_ci_high": ci_high,
            "boot_diff_samples": np.asarray(boot_diffs),
            "boot_stat_samples": np.asarray(boot_stats),
            "boot_p_samples": np.asarray(boot_ps),
        }

    return results

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
        if 'saddler_dataset' in stem:
            str_name = 'Backbone arch Saddler dataset'
        elif 'babble' in stem:
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
                    'Norm before gain model'
                ]
model_name_order_w_50Hz = model_name_order + ["50Hz cutoff"]

# set colors for human model similarity
model_color_dict = {

    'Feature-gain Model': '#0FB5AE',
    'Baseline CNN': '#808080',
    'Early-only': '#4046CA',
    'Late-only': '#F68511',
    'Norm before gain model': 'gold',
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
    if p_val >= 0.05:
        return None
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
        split1 = data.sample(frac=0.5, replace=False)
        split2 = data.drop(split1.index)
        split1 = split1.groupby(groupby_condition)[measure_string].mean().values
        split2 = split2.groupby(groupby_condition)[measure_string].mean().values
        print(split1.shape, split2.shape)
        r, p = stats.pearsonr(split1, split2)
        reliabilities[i] = r
    mean_r = np.mean(reliabilities)
    split_half_r = (2*mean_r) / (1 + mean_r)
    return split_half_r, reliabilities


# split half reliability
def split_half_reliability_trial_level(data: pd.DataFrame,
                           measure_string: Optional[str] = "measure",
                           groupby_condition: Optional[Tuple[str, list]] = ["stim_name", "measure"],
                           sortby_string: Optional[str] = "stim_name",
                           n_splits: Optional[int] = 1000,
                           tqdm: Optional[bool] = False,
                           n_to_samp: int = 1,
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
    condition_counts = data.condition.value_counts()
    to_keep = condition_counts[condition_counts >= 4].index
    data = data[data.condition.isin(to_keep)]
    conditions = data.condition.unique()
    n_conds = len(conditions)
    cond_dict = {i: cond for i, cond in enumerate(conditions)}
    reliabilities = np.zeros((n_splits, n_conds))
    if tqdm:
        from tqdm import trange
        iterable = trange(n_splits, desc="Calculating split-half reliability")
    else:
        iterable = range(n_splits)
    for i in iterable:
        split1 = data.groupby(groupby_condition).sample(n=n_to_samp, random_state=i, replace=False)
        split2 = data.drop(split1.index).groupby(groupby_condition).sample(n=n_to_samp, random_state=i, replace=False)
        for j in range(n_conds):
            cond = cond_dict[j]
            split1_cond = split1[split1.condition == cond]
            split2_cond = split2[split2.condition == cond]
            split1_cond = split1_cond.sort_values(sortby_string)[measure_string].values
            split2_cond = split2_cond.sort_values(sortby_string)[measure_string].values
            r, p = stats.pearsonr(split1_cond, split2_cond)
            reliabilities[i, j] = r

    mean_r = np.nanmean(reliabilities, axis=0) # average per condition 
    split_half_r = (2*mean_r) / (1 + mean_r)
    return split_half_r, reliabilities, cond_dict

#################################
# Color pallete for fig 2 and 6
#################################

def diotic_exp_color_palette():
    # add colors for diotic experimental conditions 
    hue_order = ['clean', '1-talker',  '2-talker',  '4-talker', 'babble'] # 'noise',  'music', 'natural scene']
    palette={}
    palette['clean'] = 'k'
    palette['Mandarin'] = 'seagreen'

    # set speech color gradient 
    speech_palette = sns.color_palette("RdPu_r")
    speech_order = hue_order[1:][::-1]

    for ix, group in enumerate(speech_order):
        palette[group] = speech_palette[ix]

    # add colors for noise conditions 
    noise_order = ['noise',  'music', 'natural scene']
    noise_palette = sns.color_palette("YlOrBr_r", n_colors=6)
    noise_order = noise_order[::-1]

    for ix, group in enumerate(noise_order):
        palette[group] = noise_palette[ix]

    # add same and different sex color palette 
    hue_order = ['Different', 'Same']
    sex_palette = dict(zip(hue_order, sns.color_palette(palette='colorblind', n_colors=10, as_cmap=False)))
    palette['Same'] = sex_palette['Same']
    palette['Different'] = 'tab:cyan'
    palette['English'] = 'tab:pink'
    palette['Mandarin'] = 'seagreen'

    return palette

