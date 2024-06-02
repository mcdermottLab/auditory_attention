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
	if bounds_from_data is not None:
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

