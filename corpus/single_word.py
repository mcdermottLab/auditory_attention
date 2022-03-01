import pandas as pd
from datasets import load_dataset # huggingface api
from tqdm import tqdm
from pathlib import Path
from os.path import join, getsize
from torch.utils.data import Dataset
import torchaudio
import torchaudio.transforms as T


# Additional (official) text src provided
# OFFICIAL_TXT_SRC = ['data/text']
# Remove longest N sentence in text
REMOVE_TOP_N_TXT = 5000000
# Default num. of threads used for loading GigaSpeech
READ_FILE_THREADS = 4
SAMPLING_RATE=16000


class SingleWordDataset(Dataset):
    def __init__(self, path, split, tokenizer, bucket_size, ascending=False):
        # Setup
        self.path = path
        self.bucket_size = bucket_size
        self.tokenizer = tokenizer
        # Load csv
        if type(split) is list:
            split = split[0]
        csv_path = join(path, f"{split}.csv")
        self.dataset = load_dataset('csv', data_files=csv_path, split='train') # train is from huggingface 
        self._len = self.dataset.num_rows
        
    def __getitem__(self, index):
        # Returns wav segment & text vs file path & text from index
        if self.bucket_size > 1:
            # Return a bucket
            index = min(self._len-self.bucket_size, index)
            bucket = dataset[index:index+self.bucket_size]
            return [(path, self.tokenizer.encode(word)) for path, word in 
                       zip(bucket['wav_path'], bucket['word'])]
        else:
            sample = self.dataset[index]
            return sample['wav_path'], self.tokenizer.encode(sample['word'])

    def __len__(self):
        return self._len


class GigaTextDataset(Dataset): # # TODO: Convert this from librispeech to gigaspeech
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
