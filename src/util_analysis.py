import scipy 
import numpy as np 

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


##################################
# model names for plotting
##################################

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
				   "word_task_v09_cue_loc_task": "Dual task model "
                  }