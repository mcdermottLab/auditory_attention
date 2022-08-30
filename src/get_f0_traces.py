import numpy as np 
from argparse import ArgumentParser
from pathlib import Path 
from tqdm.auto import tqdm
from joblib import Parallel, delayed
import librosa
import h5py
import torch 

data_path = '/om2/user/msaddler/projects/ibmHearingAid/assets/data/datasets/JSIN_v3.00/nStim_20000/2000ms/rms_0.1/noiseSNR_-10_10/stimSR_20000/reverb_none/noise_all/JSIN_all_v3/subsets/valid_RQTTZB4C3TJJVLJUWDV72TYMC7S4MNHH'

validation_h5s = Path(data_path).glob('*.h5')


class H5Dataset(torch.utils.data.Dataset):
    '''Light dataset object for reading from h5 files'''
    def __init__(self, path):
        self.file_path = path
        self.dataset = None
        with h5py.File(self.file_path, 'r', swmr=True) as file:
            self.dataset_len = len(file['sources']['signal']['signal'])
    
    def __getitem__(self,index):
        if self.dataset is None:
            self.dataset = h5py.File(self.file_path, 'r', swmr=True) # ["ndarray_data"]["signal"]
        signal = self.dataset['sources']['signal']['signal'][index]
        talker = self.dataset['sources']['signal']['speaker_int'][index]
        word = self.dataset['sources']['signal']['word_int'][index]
        return signal, talker, word

def get_f0(data_ix, dataset):
    '''Calculate f0 for voiced regions of signal and return f0, talker, and word'''
    signal, talker, word = dataset[data_ix]
    f0, voiced_flag, voiced_probs = librosa.pyin(signal,
                                             sr=20000,
                                             fmin=75,
                                             fmax=450)
    return [f0, talker, word]


def cli_main():
    # get command line arguments 
    parser = ArgumentParser()
    parser.add_argument(
    "--n_jobs",
    default=0,
    type=int,
    help="Number of CPUs for dataloader. (Default: 0)",
    )
    parser.add_argument(
        "--array_id",
        default=0,
        type=int,
        help="Slurm array task ID",
    )
    args = parser.parse_args()

    # get wanted validation set for this job
    validation_set_path = validation_h5s[args.array_id]
    # init pytorch dataset 
    dataset = H5Dataset(validation_set_path)
    # get labels 
    labels = Parallel(n_jobs=args.n_jobs)(
            delayed(get_f0)(ix, dataset) for ix in tqdm(range(len(dataset)), total=len(dataset)))

    # save labels as np array 
    all_labels = np.array(labels)
    out_name = f"/om2/user/imgriff/projects/End-to-end-ASR-Pytorch/{validation_set_path.stem}_traces.npy" 
    np.save(out_name, all_labels)


if __name__ == "__main__":
    cli_main()





