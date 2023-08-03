# Auditory Attention

This is a research project for the development of deep learning models of human audiory attention. 

## Dependencies

- Python 3
- Computing power (high-end GPU) and memory space (both RAM/GPU's RAM) is **extremely important** if you'd like to train your own model.

## ToDo

- Finish writing experiment code - but this is in github.com @rphess/SpeakerArray

## Structure

- 'attn_cue_models/' is a folder full of training checkpoints and outlogs from PyTorch Lightning. Sub folders are named for the model architecture/task and
    contain their checkpoints.

- 'binaural_eval/' is a folder containing results from evaluation scripts. Folders named after models correspond to the 19 x 19 distractor/target experiment results.

- 'confidenceScores/' is a folder containing the temperature scaling value parameters from experiments aiming to get the softmax outputs of the model to correspond to confidence scores.

- 'config/' is eponymous

- 'corpus/' contains dataset capabilities for jsin and timit, and also the binaural attention h5 file written to take batches as one index.

- 'notebooks/' contains jupyter notebooks for data exploration and model evaluation- see inside for another readme.

- 'src/' contains necessary code for the models. files starting 'spatial_attn' are the modular architecture files, 'binaural_attn' is the old model.

- The files 'configs.pkl', 'large_archs.pkl', 'loc_loc.pkl', 'mixed_cue_models.pkl', 'most_tasks_06_26.pkl', and 'train_models_08_02.pkl' are files containing lists of paths to model configurations to be used with the 'train_binaural_modular.sh' script.

- eval_binaural and eval_timit are used to output results for the speaker room and the timit dataset. Eval binaural is currently written to handle the non-modular architectures.
comments in the file will tell you how to change it for modular architectures.

- binaural_test_manifest.pdpkl is a manifest of paths and cues for 789 words from SWC with one male and one female exmaple

- speaker_room_0_elev_conditions is the 19 x 19 list used to specify the locations for speaker room evaluation in binaural_eval.py. One could make another pickle file of conditions to use with it

## Acknowledgements 


