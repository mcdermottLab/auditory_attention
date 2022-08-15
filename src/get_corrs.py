import numpy as np 
import pickle 
from scipy import stats
from tqdm.auto import tqdm


act_name = 'attn_cue_models/attn_cue_jsin_pilot_no_pretrain_norm_at_input_pos_slope_bs_64_lr_1e-4/model_output_reps.pkl'


# 17Gb array - I/O takes a little while
with open(act_name ,'rb') as f:
    activations = pickle.load(f)
    
mixture_reps = activations['mixture_reps']
fg_reps = activations['fg_reps']
bg_reps = activations['bg_reps']


## Get correlations by layer

fg_corr_results = {}
bg_corr_results = {}


n_sounds = 100

for layer, mixture_acts in tqdm(mixture_reps.items()):
    fg_acts  = fg_reps[layer]
    bg_acts = bg_reps[layer]
    
    # Calculate corr coefs for mixture and foreground
    fg_r = np.corrcoef(mixture_acts, fg_acts)
    # get coeffs of wanted samples
    fg_corr_results[layer] = np.diagonal(fg_r[:100, 100:])
    
    # Calculate corr coefs for mixture and background
    bg_r = np.corrcoef(mixture_acts, bg_acts)
    # get coeffs of background 
    bg_corr_results[layer] = np.diagonal(bg_r[:100, 100:])
    

    

out_name = 'attn_cue_jsin_pilot_no_pretrain_norm_at_input_pos_slope_bs_64_lr_1e-4.pkl'

out_dict = dict(fg_corr_results=fg_corr_results, bg_corr_results=bg_corr_results)

with open(out_name ,'wb') as f:
    pickle.dump(out_dict, f) 
    