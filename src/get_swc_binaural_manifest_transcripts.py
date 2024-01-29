
import numpy as np 
import pandas as pd 
from pathlib import Path
import pickle 
import whisper
from tqdm.auto import tqdm
import argparse
from num2words import num2words
from collections.abc import Iterable
import re 

def flatten(xs):
    for x in xs:
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
            yield from flatten(x)
        else:
            yield x

def convert_transcript(tscrpt):
    tscrpt = tscrpt.lower()
    tscrpt = re.sub(r'[^a-z0-9 ]+', '', tscrpt)
    tscrpt = [num2words(token).split('-') if token.isdigit() else token for token in tscrpt.split(' ')]
    return list(flatten(tscrpt))


def main(args):
    parent_dir = Path('/om2/user/imgriff/datasets/spatial_audio_pipeline/assets/human_attn_experiment_v00/')
    manifest = pd.read_pickle(parent_dir / 'full_eval_trial_manifest_new_fnames.pdpkl')
    out_manifest_name = parent_dir / 'full_eval_trial_manifest_new_fnames_w_transcripts.pdpkl'
    # check if the manifest already exists
    if out_manifest_name.exists() and not args.overwrite:
        print("Manifest already exists")
        return
    
    # get file names as lists
    src_fns = manifest.src_fn.to_list()
    distractor_fns = manifest.distractor_src_fn.to_list()

    print("Placing model on GPU")
    model = whisper.load_model("medium").cuda()
    options = whisper.DecodingOptions()

    batch_size = args.batch_size
    n_iters = len(src_fns) // batch_size

    transcripts = []

    print("Transcribing audio...")
    for iter in tqdm(range(n_iters)):
        target_batch = src_fns[iter*batch_size : (iter+1)*batch_size]
        # decode the audio
        target_audio = np.array([whisper.pad_or_trim(whisper.load_audio(fname)) for fname in target_batch])
        target_mel_specs = whisper.log_mel_spectrogram(target_audio).to(model.device)
        target_results = whisper.decode(model, target_mel_specs, options)

        # distrctor batch 
        distractor_batch = distractor_fns[iter*batch_size : (iter+1)*batch_size]
        distractor_audio = np.array([whisper.pad_or_trim(whisper.load_audio(fname)) for fname in distractor_batch])
        distractor_mel_specs = whisper.log_mel_spectrogram(distractor_audio).to(model.device)
        distracted_results = whisper.decode(model, distractor_mel_specs, options)

        # process text to lower case, remove punctuation, and convert digits to words
        batch_dict = {'src_fn': target_batch, 'distractor_src_fn': distractor_batch,
                    'target_transcripts': [convert_transcript(result.text) for result in target_results],
                    'distractor_transcripts': [convert_transcript(result.text) for result in distracted_results]}

        transcripts.append(batch_dict)

    all_transcripts_df = pd.concat([pd.DataFrame.from_dict(batch_dict) for batch_dict in transcripts], axis=0, ignore_index=True)
    ground_truth_df = manifest.merge(all_transcripts_df, on=['src_fn', 'distractor_src_fn'])
 
    # write out the manifest
    ground_truth_df.to_pickle(out_manifest_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--array_ix', type=int, default=0, help='array index')
    parser.add_argument('--batch_size', type=int, default = 160, help='batch size')
    parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        help="If true, will overwrite existing results",
    )
    args = parser.parse_args()
    main(args)