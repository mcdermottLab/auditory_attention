# Auditory Attention

Repository associated with publication  Ian Griffith, R. Preston Hess and Josh H. McDermott (submitted) _Optimized feature gains explain and predict successes and failures of human selective listening._

## Dependencies

- Python 3.11.5
- Pytorch 2.1+
- Pytorch lightning 2.1+
- Computing power (~4 A100 GPUs) and memory space (both 100GB RAM/ 80GB GPU memory) are necessary if you'd like to train your own model.


## Required data 
- 'attn_cue_models/' is a folder full of training checkpoints and outlogs from PyTorch Lightning. Sub folders are named for the model architecture/task and
    contain their checkpoints.

- 'demo_stimuli/' is a folder containing example audio files (as .wav files) that can be run through the model. These include cue and target excerpts produced by both a male and female talker. Below is an example demonstrating how to pre-process the audio to generate a two-talker mixture, and run the audio through the model.

- 'final_results_to_share' contains summarized data for all model and human experiments, and is used by the scripts in `notebooks/Final_Figures/` to reproduce the respective figures. 

## Structure

- 'config/' is a folder of `.yaml` configuration files specifying training data configurations, the cochlear front end, and model hyperparameters for a given model. 

- 'corpus/' contains pytorch dataset classes training and human experiment simulations.

- 'notebooks/' contains jupyter notebooks for data exploration and figure generation see inside for another readme.
    - 'notebooks/Final_Figures/' contains both `.ipynb` and `.py` files that can be used to reproduce all main and supplementary figures.
    - run `notebooks/Final_Figures/run_all_figure_gen.py` to generate all figures and run all reported statistics.     

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
import pickle 
from pathlib import Path
from src.spatial_attn_lightning import BinauralAttentionModule 
import src.audio_transforms as at
import soundfile as sf 

config_path = "config/binaural_attn/word_task_v10_main_feature_gain_config.yaml"
config = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)

# set checkpoint path
ckpt_path  =  'attn_cue_models/word_task_v10_main_feature_gain_config/checkpoints/epoch=1-step=24679-v1.ckpt'

# load model from checkpoint and freeze with .eval()
model = BinauralAttentionModule.load_from_checkpoint(checkpoint_path=ckpt_path, config=config, strict=False).eval()

# send to gpu
model = model.cuda()

# get cochleagram 
coch_gram = model.coch_gram.cuda()

# define audio transforms
SNR = 0 # signal-to-noise ratio in dB for CombineWithRandomDBSNR. Setting low and high to same value sets snr to that value
audio_transforms = at.AudioCompose([
                        at.AudioToTensor(),
                        at.CombineWithRandomDBSNR(low_snr=SNR, high_snr=SNR), 
                        at.RMSNormalizeForegroundAndBackground(rms_level=0.02),
                        at.DuplicateChannel(),
                        at.UnsqueezeAudio(dim=0),
                        ])

# Load word dictionary 
with open("./cv_800_word_label_to_int_dict.pkl", "rb") as f:
    word_to_ix_dict = pickle.load(f) 

# Map for class ix to word labels
class_ix_to_word = {v: k for k, v in word_to_ix_dict.items()}

# Load audio demo stimuli
outdir = Path("demo_stimuli")

female_cue, _ = sf.read(outdir / "female_cue.wav")
male_cue, _ = sf.read(outdir / "male_cue.wav")

female_target, _ = sf.read(outdir / "female_target_above.wav")
male_target, _ = sf.read(outdir / "male_target_about.wav" )

# use demo labels 
female_target_word = 'above'
male_target_word = 'about'

# transform audio
mixture, _ = audio_transforms(female_target, male_target) # will combine first and second signal at specified dB SNR 
female_cue, _ = audio_transforms(female_cue, None) # can pass None if not processing distractor 
male_cue, _ = audio_transforms(male_cue, None)

# get cochleagrams 
female_cue_cgram, male_cue_cgram = coch_gram(female_cue.cuda().float(), male_cue.cuda().float())
mixture_cgram, _ = coch_gram(mixture.cuda().float(), None)

# get model prediction when cueing male talker
model_logits = model(male_cue_cgram, mixture_cgram)
male_word_pred = model_logits.softmax(-1).argmax(dim=1).item()
print(f"Male cue -> True word: {male_target_word}. Predicted word: {class_ix_to_word[male_word_pred]}")
# should print "True word: about. Predicted word: about"

# get model predictions when cueing female talker in same mixture
model_logits = model(female_cue_cgram, mixture_cgram)
female_word_pred = model_logits.softmax(-1).argmax(dim=1).item()
print(f"Female cue -> True word: {female_target_word}. Predicted word: {class_ix_to_word[female_word_pred]}")
# should print "True word: above. Predicted word: above"