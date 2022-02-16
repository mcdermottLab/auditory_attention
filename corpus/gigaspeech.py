import pandas as pd
import kaldiio
from tqdm import tqdm
from pathlib import Path
from os.path import join, getsize
from joblib import Parallel, delayed
from torch.utils.data import Dataset


# Additional (official) text src provided
OFFICIAL_TXT_SRC = ['data/text']
# Remove longest N sentence in text
REMOVE_TOP_N_TXT = 5000000
# Default num. of threads used for loading GigaSpeech
READ_FILE_THREADS = 4
SAMPLING_RATE=16000


class GigaDataset(Dataset):
    def __init__(self, path, split, tokenizer, bucket_size, ascending=False):
        # Setup
        self.path = path
        self.bucket_size = bucket_size
        # Load csv
        if type(split) is list:
            split = split[0]
        csv_path = Path(path,f"data/{split}.csv")
        file_csv = pd.read_csv(csv_path, sep='\t', dtype='str') 
        # filter bad files
        file_csv = file_csv[~file_csv.isna().any(axis=1)]
        # load file segments - csv of excerpt, wav file path, eg start, eg end
        segments_path = Path(path, f"data/segments")
        segments = pd.read_csv(segments_path, header=None,
                       names=['wav_filename', 'speaker', 'start', 'end'],
                       sep='\t')
        # map of speaker to wav path
        self.wavscp = kaldiio.load_scp(Path(path,'data','wav.scp').as_posix())

        # merge file csv with segmnet onsets & offsets
        times = segments[['start', 'end']][segments.wav_filename.isin(file_csv.wav_filename)]
        file_csv = pd.concat([file_csv, times.set_index(file_csv.index)], axis=1)
        # Tokenize transcript
        file_csv['transcript'] = file_csv['transcript'].apply(tokenizer.encode)
        # Sort dataset by text length & make attribute
        self.file_csv = file_csv.sort_values(by='transcript', key=lambda x: x.str.len(), ascending=ascending)

    def get_wav_from_item(self, item):
        # Parses contents of csv item and wavscp to return wav and transcript
        name = item.wav_filename
        speaker = item.speaker
        start_time = int(float(item.start) * SAMPLING_RATE)
        end_time = int(float(item.end) * SAMPLING_RATE)
        wav = self.wavscp[speaker][1][start_time:end_time]
        text = item.transcript
        return name, wav, text

    def __getitem__(self, index):
        # Returns wav segment & text vs file path & text from index
        if self.bucket_size > 1:
            # Return a bucket
            index = min(len(self.file_csv)-self.bucket_size, index)
            return [(self.get_wav_from_item(item[1])) for item in
                    self.file_csv.iloc[index:index+self.bucket_size].iterrows()]
        else:
            item = self.file_csv.iloc[index]
            name, wav, text = self.get_wav_from_item(item)
            return name, wav, text

    def __len__(self):
        return len(self.file_csv)


class GigaTextDataset(Dataset): # # TODO: Make
    def __init__(self, path, split, tokenizer, bucket_size):
        # Setup
        self.path = path
        self.bucket_size = bucket_size
        self.encode_on_fly = False
        read_txt_src = []

        # List all wave files
        file_list, all_sent = [], []

        for s in split:
            if s in OFFICIAL_TXT_SRC:
                self.encode_on_fly = True
                with open(join(path, s), 'r') as f:
                    all_sent += f.readlines()
            file_list += list(Path(join(path, s)).rglob("*.flac"))
        assert (len(file_list) > 0) or (len(all_sent)
                                        > 0), "No data found @ {}".format(path)

        # Read text
        text = Parallel(n_jobs=READ_FILE_THREADS)(
            delayed(read_text)(str(f)) for f in file_list)
        all_sent.extend(text)
        del text

        # Encode text
        if self.encode_on_fly:
            self.tokenizer = tokenizer
            self.text = all_sent
        else:
            self.text = [tokenizer.encode(txt) for txt in tqdm(all_sent)]
        del all_sent

        # Read file size and sort dataset by file size (Note: feature len. may be different)
        self.text = sorted(self.text, reverse=True, key=lambda x: len(x))
        if self.encode_on_fly:
            del self.text[:REMOVE_TOP_N_TXT]

    def __getitem__(self, index):
        if self.bucket_size > 1:
            index = min(len(self.text)-self.bucket_size, index)
            if self.encode_on_fly:
                for i in range(index, index+self.bucket_size):
                    if type(self.text[i]) is str:
                        self.text[i] = self.tokenizer.encode(self.text[i])
            # Return a bucket
            return self.text[index:index+self.bucket_size]
        else:
            if self.encode_on_fly and type(self.text[index]) is str:
                self.text[index] = self.tokenizer.encode(self.text[index])
            return self.text[index]

    def __len__(self):
        return len(self.text)
