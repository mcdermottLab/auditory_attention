# Auditory Attention

This repository accompanies Ian Griffith, R. Preston Hess, and Josh H. McDermott, _Optimized feature gains explain and predict successes and failures of human selective listening_.  
It contains the trained attention models, curated human-participant datasets, demo stimuli, notebooks, and evaluation scripts needed to reproduce every figure and statistic in the manuscript. The sections below walk new users through the layout, dependencies, and common workflows.

## Repository map

- `src/` – PyTorch Lightning modules, cochlear front-end implementations, audio transforms, and utilities shared across training and evaluation.
- `config/` – YAML files describing model architectures, datasets, and hyperparameters.
- `corpus/` – Dataset/dataloader definitions for model training and behavioral simulations.
- `notebooks/` – Jupyter notebooks for exploratory analysis and figure generation.
  - `notebooks/Final_Figures/` contains both `.ipynb` and `.py` counterparts for every main and supplementary figure. Run `python notebooks/Final_Figures/run_all_figure_gen.py` to regenerate the full figure suite and associated statistics.
- `participant_data/` – Clean CSV/PKL tables for human experiments (per-trial logs, summary stats, ANOVA inputs, metadata). These are the authoritative data sources consumed by the notebooks.
- `attn_cue_models/` – Pretrained checkpoints and PyTorch Lightning logs for the feature-gain models analyzed in the paper.
- `demo_stimuli/` – Male/female cue-target `.wav` files plus mixtures so you can audition the models immediately.
- `final_results_to_share/` – Aggregated model + human metrics used by the figure scripts.
- `eval_*.py` – Standalone scripts that reproduce each behavioral experiment (see “Running analyses” below).
- `*.sh` – SLURM-ready job scripts showing how we ran the corresponding Python programs on MIT OpenMind.

## Included data directories

- `attn_cue_models/` ships with checkpoints and training logs organized by architecture/task (e.g., `word_task_v10_main_feature_gain_config`). Each directory contains the `.ckpt` files referenced throughout the manuscript.
- `demo_stimuli/` provides the example `.wav` files used in the quick-start code below (male/female cues, targets, and mixtures).
- `participant_data/` contains curated experiment tables: raw participant thresholds, per-trial CSVs, summary pickles for ANOVAs, and metadata spreadsheets.
- `final_results_to_share/` offers precomputed aggregates for every model/human comparison, allowing the notebooks to render figures without recomputing heavy simulations.

## Getting started

1. **Install dependencies**
   - Python 3.11.5
   - PyTorch 2.1+
   - PyTorch Lightning 2.1+
   - Additional packages listed in `requirements.txt` (recommended: create a conda env and `pip install -r requirements.txt`)
2. **Hardware expectations**  
   Training fresh models mirrors the paper’s setup (~4×A100 GPUs and ~100 GB host / 80 GB device RAM). Figure generation and evaluation can run on a single GPU with modest memory.
3. **Verify tracked assets**  
   After cloning, confirm that `attn_cue_models/`, `demo_stimuli/`, `participant_data/`, and `final_results_to_share/` are present. If not, re-run `git lfs pull` (when applicable) or download the shared data bundle.

## Running analyses

- **Regenerate the full figure suite**
  ```
  python notebooks/Final_Figures/run_all_figure_gen.py
  ```
  Outputs land in `notebooks/Final_Figures/all_figures_output/` and include the statistics reported in the manuscript.
- **Per-experiment simulations**
  - `eval_swc_mono_stim.py` – Experiment 1
  - `eval_swc_popham_2024.py` – Experiment 2
  - `eval_texture_backgrounds.py` – Experiment 3
  - `eval_symmetric_distractors.py` – Arbitrary spatial configurations plus Experiments 4–5
  - `eval_sim_array_threshold_experiment_v02.py` – Experiment 6 (thresholds)
  - `eval_sim_array_spotlight_experiment_v02.py` – Experiment 7 (spotlight task)
  - `get_acts_for_tuning_and_selection_analysis.py` – Activation dumps for Figure 5 / Supplementary Figure 5
- **Cluster execution**  
  Use the `.sh` scripts (e.g., `run_unit_tuning_anova_parallel.sh`) as templates for your scheduler; they capture the exact resource settings we used on OpenMind.

## Quick-start: load a checkpoint and run the demo stimuli
```python
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
```

This example relies entirely on tracked assets (`config/`, `attn_cue_models/`, `demo_stimuli/`, `cv_800_word_label_to_int_dict.pkl`). After confirming it runs end-to-end, you can swap in your own stimuli, adjust the audio transforms, or fine-tune the models with different configs. For deeper dives, inspect the notebooks in `notebooks/Final_Figures/` or the evaluation scripts listed earlier.