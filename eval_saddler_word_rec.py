# Following is originally copied from PyTorch RNN-T ASR Example:
# https://github.com/pytorch/audio/tree/820b383b3b21fc06e91631a5b1e6ea1557836216/examples/asr/librispeech_emformer_rnnt

import pathlib
from argparse import ArgumentParser
import yaml
import pickle
import torch 
import numpy as np 
import pandas as pd

from tqdm.auto import tqdm
from pytorch_lightning import seed_everything
from src.attn_tracking_lightning import AttentionalTrackingModule
from src.spatial_attn_lightning import BinauralAttentionModule 
from corpus.saddler_word_rec import SaddlerSWCWordRecTest
import src.audio_transforms as at
import scipy.stats as stats
import h5py
import soxr



seed_everything(1)


def run_eval(args):

    # get test conds
    with open(args.test_manifest, 'rb') as f:
        eval_conditions = pickle.load(f)
        condition, snr = eval_conditions[args.array_id]

    model_name = pathlib.Path(args.config).stem
    checkpoint_path = args.ckpt_path
    config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
    print(f"Evaluating {model_name} on {condition} at {snr}db SNR")
    print(f"Loading model from {checkpoint_path}")
    
    # load model 
    if 'binaural' in model_name:
        module = BinauralAttentionModule
        label_type = 'CV'
        sr = 50_000
    else:
        module = AttentionalTrackingModule
        config['data']['audio']['rep_kwargs']['center_crop'] = True
        config['data']['audio']['rep_kwargs']['out_dur'] = 2


        label_type = "WSN"
        sr = 20_000
    
    # set audio transforms
    audio_config = config['data']['audio']
    IIR_COCH = not audio_config['rep_kwargs']['rep_on_gpu']

    if IIR_COCH:
        audio_transforms = at.AudioCompose([
                at.AudioToTensor(),
                at.CombineWithRandomDBSNR(low_snr=snr, high_snr=snr), 
                at.RMSNormalizeForegroundAndBackground(rms_level=0.1),
                at.UnsqueezeAudio(dim=0),
                at.AudioToAudioRepresentation(**audio_config),
            ])
    else:
        audio_transforms = at.AudioCompose([
                    at.AudioToTensor(),
                    at.CombineWithRandomDBSNR(low_snr=snr, high_snr=snr), 
                    at.RMSNormalizeForegroundAndBackground(rms_level=0.1),
                    at.UnsqueezeAudio(dim=0),
            ])  

    # load and freeze model
    model = module.load_from_checkpoint(checkpoint_path=checkpoint_path, config=config).eval().cuda()

    dataset = SaddlerSWCWordRecTest(manifest_path=args.foreground_stim_path,
                                    bg_stim_path=args.background_stim_path,
                                    condition=condition,
                                    label_type=label_type,
                                    sr=sr)

    def collate_fn(batch):
        #apply transforsms to batch
        cues = torch.stack([audio_transforms(cue, None)[0] for cue, _, _, _ in batch])
        mixtures = torch.stack([audio_transforms(fg,bg)[0] for _, fg, bg, _ in batch])
        labels = torch.tensor([label for _, _, _, label in batch]).type(torch.LongTensor)
        return cues, mixtures, labels

    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=1,
                                             shuffle=False,
                                             collate_fn=collate_fn,
                                             num_workers=args.n_jobs)


    new_room_manifest = pd.read_pickle('/om2/user/msaddler/spatial_audio_pipeline/assets/brir/mit_bldg46room1004/manifest_brir.pdpkl')
    only14_manifest = new_room_manifest[new_room_manifest['src_dist'] == 1.4]
    df_row = only14_manifest[(only14_manifest['src_azim'] == 0) & (only14_manifest['src_elev'] == 0)]
    h5_fn = f'/om2/user/msaddler/spatial_audio_pipeline/assets/brir/mit_bldg46room1004/room000{df_row["index_room"].values[0]}.hdf5'
    index_brir = df_row['index_brir'].values[0]
    sr_src = df_row['sr'].values[0]
    with h5py.File(h5_fn, 'r') as f:
        brir = f['brir'][index_brir]
    sr = 50000
    brir = soxr.resample(brir.astype(np.float32), sr_src, sr)
    brir = torch.from_numpy(brir)
    brir = torch.flip(brir, dims=[0])

    def mass_spatialize(words, ir):
        """Uses pytorch to convolve all sounds in words with 2 channel IR given in ir"""
        n_words = words.shape[0]
        words_padded = [torch.nn.functional.pad(word, (ir.shape[0] - 1, 0)) for word in words]
        ir = ir.T.unsqueeze(1)
        words_padded = torch.stack(words_padded)
        spatialized = torch.nn.functional.conv1d(words_padded.view(n_words, 1, -1).cuda(), ir.cuda()).cuda()
        return spatialized

    # run eval loop
    results = []

    for batch in tqdm(dataloader):
        cue, mixture, word = batch
        # to device 
        cue = cue.cuda()
        mixture = mixture.cuda()
        
        logits = model(cue, mixture)
        preds = logits.softmax(dim=-1).argmax(dim=-1).cpu().detach()
        results.extend(word == preds)

    
    res_err = stats.sem(results)
    res = np.mean(results)

    res_dict = {"acc": res, "std_err": res_err}
    out_dir = args.exp_dir / model_name 
    # make dir if it doesn't exist
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{model_name}_{condition}_{snr}_eval_results.pkl", 'wb') as f:
        pickle.dump(res_dict, f)

    print(f"Eval results: {res} +/- {res_err}")

def cli_main():
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default="", help='Path to experiment config.')
    parser.add_argument(
        "--exp_dir",
        default=pathlib.Path("./exp"),
        type=pathlib.Path,
        help="Directory to save checkpoints and logs to. (Default: './exp')",
    )
    parser.add_argument(
        "--foreground_stim_path",
        default=pathlib.Path("//om2/user/imgriff/datasets/spatial_audio_pipeline/assets/human_experiment_v00/cue_and_target_manifest.pdpkl"),
        type=pathlib.Path,
        help="Path where background stimuli are saved",
    )
    parser.add_argument(
        "--background_stim_path",
        default=pathlib.Path("/om2/user/msaddler/spatial_audio_pipeline/assets/human_experiment_v00/"),
        type=pathlib.Path,
        help="Path where background stimuli are saved",
    )
    parser.add_argument(
        "--ckpt_path",
        default=pathlib.Path("./exp"),
        type=pathlib.Path,
        help="path to checkpoint (Default: './exp')",
    )
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
    parser.add_argument(
        "--test_manifest",
        default=pathlib.Path("saddler_test_condition_dict.pkl"),
        type=pathlib.Path,
        help="Path to test manifest",
    )    
    args = parser.parse_args()

    run_eval(args)


if __name__ == "__main__":
    cli_main()
