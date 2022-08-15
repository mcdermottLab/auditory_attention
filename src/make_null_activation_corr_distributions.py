import numpy as np 
import pickle 
from tqdm.auto import tqdm

import argparse
import os 



if __name__ == '__main__':
    
    #########PARSE THE ARGUMENTS FOR THE FUNCTION#########
    parser = argparse.ArgumentParser(description='Input the arguments for the null distribution generation')
    parser.add_argument('-N', '--NUMNULL', metavar='--N', type=int, default=10000, help='Number of samples to generate for the null distribution')
    parser.add_argument('-R', '--RANDOMSEED', metavar='--R', type=int, default=0, help='random seed to use for synthesis')
    args=parser.parse_args()



    out_name = './attn_cue_models/attn_cue_jsin_pilot_no_pretrain_norm_at_input_pos_slope_bs_64_lr_1e-4/model_output_reps.npz'

    activations = np.load(out_name, allow_pickle=True)

    # Unpack - I/O takes a little while    
    mixture_reps = activations['mixture_reps'].item()
    fg_reps = activations['fg_reps'].item()

    save_features_dir = './null_activation_corrs'

    if not os.path.isdir(save_features_dir):
        os.mkdir(save_features_dir)


    filename = os.path.join(save_features_dir, f'mixture_null_model_corrs_{args.RANDOMSEED}.pkl')

    ## Get correlations by layer


    np.random.seed(args.RANDOMSEED)

    n_sounds = 100
    
    null_cor_results_dict = {layer:np.zeros((args.NUMNULL, n_sounds)) for layer in mixture_reps.keys()}

    
    for n in tqdm(range(args.NUMNULL)):

        for layer, mixture_acts in tqdm(mixture_reps.items(), leave=False):

            fg_ixs  = np.random.choice(n_sounds, size=n_sounds, replace=True)

            fg_acts  = fg_reps[layer][fg_ixs,:]
            
            # Calculate corr coefs for mixture and foreground
            null_r = np.corrcoef(mixture_acts, fg_acts)
            # fg_r = spearmanr(mixture_acts, fg_acts, axis=1)[0]
            # get coeffs of wanted samples
            null_cor_results_dict[layer][n,:] = np.diagonal(null_r[:n_sounds, n_sounds:])
        
        
    with open(filename ,'wb') as f:
        pickle.dump(null_cor_results_dict, f, protocol=pickle.HIGHEST_PROTOCOL) 
        