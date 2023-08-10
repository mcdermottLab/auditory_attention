# Binaural_cue_voice_timit.ipynb

Copied over from Ian's notebooks. I beleive it was tests to see how to test timit dataset on current models.

# binaurual_dataset.ipynb

Opening and playing sounds from the pre-generated data set. Also shows how to dump a list of configs into a pkl file.

# check_BRIRs.ipynb

Plotting and listening to exisiting BRIRs, and seeing how to convolve them with scenes. There are new paths to IRs now with Mark's new data set.

# check_softmax_outputs.ipynb

calculate and plot different confidence metrics compared to model accuracy. This also includes tests that lead to temperature scaling tests and results.

# create_validation_examples.ipynb

Early tests leading to evaluation scripts. Resulted in first_376.h5 which is an h5 files containing 376 words spatialized at each speaker location (old room brirs)

# generate_test_set.ipynb

first section created manifest with 789 words in vocabulary from SWC with one male and one female example and a cue for each. The second part spatialized these on the fly,
which were tests for what eventually became the evaluation script.

# max_confusion.ipynb

similar to check_softmax_outputs, but was aiming to test which words produced low confidence. Did not seem to get there yet.

# pipeline_03_20_2023.ipynb

Code that lead to training data files of spatialized sounds- incomplete here but finished by Ian over early summer break.

# test_modular_architecture.ipynb

loads in both modular and static architectures with same random weights in order to check if outputs are the same. They are.

# test_plots.ipynb

Probably most important as of right now (08/03/2023). This takes the evaluation outputs and graphs matrices of accuracy and confusions.

# test_trainloader.ipynb

this made the trainloader index into whole batches instead of grabbing them one at a time. Code from this is now implemented in the dataset class.

# validate_location_model.ipynb

Tested the location cue location task model as a sanity check and includes functions that code results of azimuth and elevation error.