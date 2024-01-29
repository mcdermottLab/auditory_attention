
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
    map_path = Path('/om2/user/imgriff/projects/Auditory-Attention/human_saddler_attn_expmt_cond_map.pkl')
    with open(map_path, 'rb') as handle:
        stim_cond_map = pickle.load(handle)

    stim_cond_map = {f"condition_{k:02}": v for k,v in stim_cond_map.items()}

    cond_dir = f"condition_{args.array_ix:02}"
    cond, _ = stim_cond_map[cond_dir]
    parent_dir = Path("/om/user/imgriff/datasets/human_word_rec_SWC_2023/sounds/")
    manifes_path = parent_dir / cond_dir / 'manifest.pdpkl'
    out_manifest_path = parent_dir / cond_dir / 'manifest_w_transcripts.pdpkl'
    # check if the manifest already exists
    if out_manifest_path.exists() and not args.overwrite:
        print("Manifest already exists")
        return
    
    manifest = pd.read_pickle(manifes_path)

    target_fns = manifest.target_fn.to_list()
    if 'distractor_fn' in manifest.columns and cond != "4-talker":
        distractor_fns = manifest.distractor_fn.to_list()
        print("Getting distractor transcripts too")
    else:
        distractor_fns = None

    print("Placing model on GPU")
    model = whisper.load_model("medium").cuda()
    options = whisper.DecodingOptions()

    batch_size = args.batch_size
    n_iters = np.ceil(len(target_fns) / batch_size).astype(int)

    transcripts = []

    
    print("Transcribing audio...")
    for iter_ix in tqdm(range(n_iters)):
        start = iter_ix * batch_size
        end = start + batch_size
        if end > len(target_fns):
            end = len(target_fns)
        target_batch = target_fns[start : end]
        # decode the audio
        target_audio = np.array([whisper.pad_or_trim(whisper.load_audio(fname)) for fname in target_batch])
        target_mel_specs = whisper.log_mel_spectrogram(target_audio).to(model.device)
        target_results = whisper.decode(model, target_mel_specs, options)

        # distrctor batch 
        if distractor_fns is not None:
            distractor_batch = distractor_fns[start : end]
            distractor_audio = np.array([whisper.pad_or_trim(whisper.load_audio(fname)) for fname in distractor_batch])
            distractor_mel_specs = whisper.log_mel_spectrogram(distractor_audio).to(model.device)
            distracted_results = whisper.decode(model, distractor_mel_specs, options)

        # decode the audio

        # process text to lower case, remove punctuation, and convert digits to words
        if distractor_fns is not None:
            batch_dict = {'target_fn': target_batch, 'distractor_fn': distractor_batch,
                        'target_transcripts': [convert_transcript(result.text) for result in target_results],
                        'distractor_transcripts': [convert_transcript(result.text) for result in distracted_results]}
        else:
            batch_dict = {'target_fn': target_batch,
                        'target_transcripts': [convert_transcript(result.text) for result in target_results]}
            
        transcripts.append(batch_dict)

    all_transcripts_df = pd.concat([pd.DataFrame.from_dict(batch_dict) for batch_dict in transcripts], axis=0, ignore_index=True)
    if distractor_fns is not None:
        ground_truth_single_dist_df = manifest.merge(all_transcripts_df, on=['target_fn', 'distractor_fn'])
    else:
        ground_truth_single_dist_df = manifest.merge(all_transcripts_df, on=['target_fn'])
    # write out the manifest
    out_manifest_path = parent_dir / cond_dir / 'manifest_w_transcripts.pdpkl'
    ground_truth_single_dist_df.to_pickle(out_manifest_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--array_ix', type=int, help='array index')
    parser.add_argument('--batch_size', type=int, default = 160, help='batch size')
    parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        help="If true, will overwrite existing results",
    )
    args = parser.parse_args()
    main(args)