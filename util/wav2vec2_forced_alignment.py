import torch
import torchaudio
import torchaudio.transforms as T


print(torch.__version__)
print(torchaudio.__version__)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


import argparse
import logging
from dataclasses import dataclass
import re
from tqdm import tqdm
from pathlib import Path
import pandas as pd

torch.random.manual_seed(0)

##############################
#        Data classes 
##############################

@dataclass
class Point:
    token_index: int
    time_index: int
    score: float


# Merge the labels
@dataclass
class Segment:
    label: str
    start: int
    end: int
    score: float

    def __repr__(self):
        return f"{self.label}\t({self.score:4.2f}): [{self.start:5d}, {self.end:5d})"

    @property
    def length(self):
        return self.end - self.start

@dataclass
class Alignment:
    label: str
    start: int
    end: int
    score: float

    def __repr__(self):
        return f"{self.label}\t({self.score:4.2f}): [{self.start:.4f}, {self.end:.4f})"

    @property
    def length(self):
        return self.end - self.start

    
##############################
#    Alignment functions 
##############################
    
# gets character-level trellis
def get_trellis(emission, tokens, blank_id=0):
    num_frame = emission.size(0)
    num_tokens = len(tokens)

    # Trellis has extra diemsions for both time axis and tokens.
    # The extra dim for tokens represents <SoS> (start-of-sentence)
    # The extra dim for time axis is for simplification of the code.
    trellis = torch.empty((num_frame + 1, num_tokens + 1))
    trellis[0, 0] = 0
    trellis[1:, 0] = torch.cumsum(emission[:, 0], 0)
    trellis[0, -num_tokens:] = -float("inf")
    trellis[-num_tokens:, 0] = float("inf")

    for t in range(num_frame):
        trellis[t + 1, 1:] = torch.maximum(
            # Score for staying at the same token
            trellis[t, 1:] + emission[t, blank_id],
            # Score for changing to the next token
            trellis[t, :-1] + emission[t, tokens],
        )
    return trellis

    
def backtrack(trellis, emission, tokens, blank_id=0):
    # Note:
    # j and t are indices for trellis, which has extra dimensions
    # for time and tokens at the beginning.
    # When referring to time frame index `T` in trellis,
    # the corresponding index in emission is `T-1`.
    # Similarly, when referring to token index `J` in trellis,
    # the corresponding index in transcript is `J-1`.
    j = trellis.size(1) - 1
    t_start = torch.argmax(trellis[:, j]).item()

    path = []
    for t in range(t_start, 0, -1):
        # 1. Figure out if the current position was stay or change
        # Note (again):
        # `emission[J-1]` is the emission at time frame `J` of trellis dimension.
        # Score for token staying the same from time frame J-1 to T.
        stayed = trellis[t - 1, j] + emission[t - 1, blank_id]
        # Score for token changing from C-1 at T-1 to J at T.
        changed = trellis[t - 1, j - 1] + emission[t - 1, tokens[j - 1]]

        # 2. Store the path with frame-wise probability.
        prob = emission[t - 1, tokens[j - 1] if changed > stayed else 0].exp().item()
        # Return token index and time index in non-trellis coordinate.
        path.append(Point(j - 1, t - 1, prob))

        # 3. Update the token
        if changed > stayed:
            j -= 1
            if j == 0:
                break
    else:
        raise ValueError("Failed to align")
    return path[::-1]


def merge_repeats(path, transcript):
    i1, i2 = 0, 0
    segments = []
    while i1 < len(path):
        while i2 < len(path) and path[i1].token_index == path[i2].token_index:
            i2 += 1
        score = sum(path[k].score for k in range(i1, i2)) / (i2 - i1)
        segments.append(
            Segment(
                transcript[path[i1].token_index],
                path[i1].time_index,
                path[i2 - 1].time_index + 1,
                score,
            )
        )
        i1 = i2
    return segments


# Merge words
def merge_words(segments, separator="|"):
    words = []
    i1, i2 = 0, 0
    while i1 < len(segments):
        if i2 >= len(segments) or segments[i2].label == separator:
            if i1 != i2:
                segs = segments[i1:i2]
                word = "".join([seg.label for seg in segs])
                score = sum(seg.score * seg.length for seg in segs) / sum(seg.length for seg in segs)
                words.append(Segment(word, segments[i1].start, segments[i2 - 1].end, score))
            i1 = i2 + 1
            i2 = i1
        else:
            i2 += 1
    return words


def get_transcript(transcript):
    transcript = '|'.join([re.sub("[^\w\d'\s]+",'', word) for word in transcript.upper().split(' ')])
    return transcript


##############################
#    Main job function 
##############################


def main(args):
     
    # Import pre-trained wav2vec2 model 
    bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
    model = bundle.get_model().to(device)
    labels = bundle.get_labels()
    # model label to ix dictionary 
    dictionary = {c: i for i, c in enumerate(labels)}
    print(device)
    # define resampler to convert audio to sampling rate of wav2vec2
    resample_rate = bundle._sample_rate
    resampler = T.Resample(args.source_sample_rate, resample_rate, dtype=torch.float32)
    
    # get dataset 
    pd_path = Path('/om2/data/public/mozilla-CommonVoice-9.0/cv-corpus-9.0-2022-04-27/en/')
    
    commonvoice_path = Path('/scratch2/weka/mcdermott/imgriff/datasets/commonvoice_9/en/')
    mp3_path = commonvoice_path / 'clips'
    # get manifest of all valid files:
    tsv_path = pd_path / 'validated.tsv'

    valid_df = pd.read_csv(tsv_path, sep='\t')
    valid_df = valid_df[~pd.isna(valid_df.sentence)]
    valid_df['origin_index'] = valid_df.index
    
    # get ix for job split
    start_ix = args.array_ix * args.batch_size
    stop_ix = start_ix + args.batch_size
    if stop_ix > len(valid_df):
        stop_ix = len(valid_df)
    
    # get slice for job split
    valid_df = valid_df.iloc[start_ix : stop_ix]
    
    # pair down to dict with wanted columns for fast iteration
    valid_meta = valid_df[['path', 'sentence', 'client_id', 'gender','origin_index']].to_dict('records')
    
    # don't track gradients
    with torch.inference_mode():
        # iter over examples
        for ix, example in tqdm(enumerate(valid_meta), total=len(valid_meta)):
            try:
                # load mp3 and process transcript
                speech_file = mp3_path / example['path']
                transcript = example['sentence']
                transcript = get_transcript(transcript)
            
                # read mp3 and process through wav2vec2
                waveform, wav_sr = torchaudio.load(speech_file)
                waveform = resampler(waveform)
                emissions, _ = model(waveform.to(device))
                # get model emissions
                emissions = torch.log_softmax(emissions, dim=-1)
                emission = emissions[0].cpu().detach()
            
                # forced alignment here
                tokens = [dictionary[c] for c in transcript]
                trellis = get_trellis(emission, tokens)
                path = backtrack(trellis, emission, tokens)
                segments = merge_repeats(path, transcript)
                word_segments = merge_words(segments)
            
                # convert words from trellis frames to seconds 
                ratio = waveform.size(1) / (trellis.size(0) - 1)
                word_alignment = []
                for i in range(len(word_segments)):
                    word = word_segments[i]
                    x0 = int(ratio * word.start) /  bundle.sample_rate
                    x1 = int(ratio * word.end) /  bundle.sample_rate
                    word_alignment.append(Alignment(word.label, x0, x1, word.score))
                # update meta dict for this eg with alignment 
                valid_meta[ix]['alignment'] = word_alignment
            except Exception as e:
                print(e, f"on step {ix}")
                valid_meta[ix]['alignment'] = np.nan
                continue 
    # Save as pandas dataframe
    valid_meta = valid_meta[valid_meta.alignment.notna()]
    alignment_data = pd.DataFrame.from_records(valid_meta)
    out_path = commonvoice_path / 'alignment_dfs' 
    out_path.mkdir(parents=True, exist_ok=True)
    out_name = out_path / f"alignment_split_{args.array_ix:03}.pdpkl"
    
    print(f"Saving stimuli to: {out_name.as_posix()}") 
    
    alignment_data.to_pickle(out_name) 

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        "Utility script to generate word alignments audio files using wav2vec2.")
#     parser.add_argument("--audio_file_dir", type=Path, required=True)
#     parser.add_argument("--output_file", required=True)
    parser.add_argument("--array_ix",
                        default=0,
                        type=int,
                        help="SLURM job array ix.",
                        )
    parser.add_argument("--batch_size",
                    default=15500,
                    type=int,
                    help="Number of files to proccess in job.",
                    )
    parser.add_argument("--source_sample_rate",
                default=32000,
                type=int,
                help="Sampling rate of audio files getting aligned",
                )
    args = parser.parse_args()



    main(args)
