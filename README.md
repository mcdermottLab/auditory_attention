# Auditory Attention

Repository associated with publication  Ian Griffith, R. Preston Hessand Josh H. McDermott (in press) _Optimized feature gains explain and predict successes and failures of human selective listening._

## Dependencies

- Python 3.12
- Pytorch 2.1+
- Pytorch lightning 2.4
- Computing power (~4 A100 GPUs) and memory space (both 100GB RAM/ 80GB GPU memory) are necessary if you'd like to train your own model.


## Structure

- 'attn_cue_models/' is a folder full of training checkpoints and outlogs from PyTorch Lightning. Sub folders are named for the model architecture/task and
    contain their checkpoints.

- 'config/' is a folder of `.yaml` configuration files specifying training data configurations, the cochlear front end, and model hyperparameters for a given model. 

- 'corpus/' contains pytorch dataset classes training and human experiment simulations.

- 'notebooks/' contains jupyter notebooks for data exploration and figure generation see inside for another readme.
    - 'notebooks/Final_Figures/' contains both `.ipynb` and `.py` files that can be used to reproduce all main and supplementary figures.
    - run `notebooks/Final_Figures/run_all_figure_gen.py` to generate all figures and run all reported statistics. 

- 'final_results_to_share' contains summarized data for all model and human experiments, and is used by the scripts in `notebooks/Final_Figures/` to reproduce the respective figures. 

- 'src/' contains necessary code for the models and stimuli generation.

- evaluation files are in the main directory level.
    - `eval_swc_mono_stim.py` is used to simulate experiment 1.
    - `eval_swc_popham_2024.py` is used to simulate experiment 2.
    - `eval_texture_backgrounds.py` is used to simulate experiment 3.
    - `eval_symmetric_distractors.py` is a modular script. It is used to simulate all arbitrary spatial configurations (figure 4a; supplementary figures 2-4), and simulate experiments 4 and 5.
    - `eval_sim_array_threshold_experiment_v02.py` is used to simulate experiment 6. 
    - `eval_sim_array_spotlight_experiment_v02.py` is used to simulate experiment 7. 
    - `get_acts_for_tuning_and_selection_analysis.py` is used to obtain model activations for the stage of selection analysis (figure 5 and supplementary figure 5). 

- all `*.sh` scripts show examples of how corresponding `.py` scripts were exectued on the OpenMind compute cluster, indluding compute resource requirments.


## Requirements 

Create a conda environment including the packages in requirements.txt to run model training, evaluation and plotting scripts. 

## Snippet for loading and running model:
```
import yaml
import src.spatial_attn_lightning as binaural_lightning 
import src.audio_transforms as at

config_path = "config/binaural_attn/word_task_v10_main_feature_gain_config.yaml"
config = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)

# after downloading checkpoints 
ckpt_path  = `attn_cue_models/word_task_v10_main_feature_gain_config/checkpoints/epoch=1-step=24679-v1.ckpt`

# load model from checkpoint and freeze with .eval()
model = binaural_lightning.load_from_checkpoint(checkpoint_path=ckpt_path, config=config, strict=False).eval()

# send to gpu
model = model.cuda()

# get cochleagram 
coch_gram = model.coch_gram.cuda()

# can load audio as desired. Audio should be 2 seconds at 44.1 kHz sampling rate.
cue, mixture = ... 

# if mono, copy signals to 2 channels 
mono_2_stereo = at.DuplicateChannel()
cue, mixture = mono_2_stereo(cue, mixture)

# get cochleagrams of cue and mixture
cue, mixture = coch_gram(cue.cuda(), mixture.cuda())

# model forward takes cue (can be None), mixture, and mask_ix (typically will none)
logits = model(cue, mixture, None)
```

