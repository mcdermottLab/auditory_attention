# Following is originally copied from PyTorch RNN-T ASR Example:
# https://github.com/pytorch/audio/tree/820b383b3b21fc06e91631a5b1e6ea1557836216/examples/asr/librispeech_emformer_rnnt

import pathlib
from argparse import ArgumentParser
import yaml
import pickle
import torch 
import numpy as np 
from tqdm.auto import tqdm
from pytorch_lightning import seed_everything
from src.attn_tracking_lightning import AttentionalTrackingModule
from src.spatial_attn_lightning import BinauralAttentionModule 
from corpus.swc_mono_test import SWCMonoTestSet
import src.audio_transforms as at
import scipy.stats as stats


seed_everything(1)


def run_eval(args):

    model_name = pathlib.Path(args.config).stem
    checkpoint_path = args.ckpt_path
    config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
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
                # at.CombineWithRandomDBSNR(low_snr=snr, high_snr=snr), 
                at.RMSNormalizeForegroundAndBackground(rms_level=0.1),
                at.UnsqueezeAudio(dim=0),
                at.AudioToAudioRepresentation(**audio_config),
            ])
    else:
        audio_transforms = at.AudioCompose([
                    at.AudioToTensor(),
                    # at.CombineWithRandomDBSNR(low_snr=snr, high_snr=snr), 
                    at.RMSNormalizeForegroundAndBackground(rms_level=0.1),
                    at.UnsqueezeAudio(dim=0),
            ])  

    # load and freeze model
    model = module.load_from_checkpoint(checkpoint_path=checkpoint_path, config=config).eval().cuda()

    dataset = SWCMonoTestSet(stim_path=args.stim_path,
                            cond_ix=args.array_id,
                            model_sr=sr,
                            label_type=label_type)
    
    condition, snr = dataset.stim_cond_map[args.array_id]
    print(f"Evaluating {model_name} on {condition} at {snr}db SNR")

    def collate_fn(batch):
        #apply transforsms to batch
        cues = torch.stack([audio_transforms(cue, None)[0] for cue, _, _ in batch])
        mixtures = torch.stack([audio_transforms(mix, None)[0] for _, mix,  _ in batch])
        labels = torch.tensor([label for _, _, label in batch]).type(torch.LongTensor)
        return cues, mixtures, labels

    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=1,
                                             shuffle=False,
                                             collate_fn=collate_fn,
                                             num_workers=args.n_jobs)

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
        "--stim_path",
        default=pathlib.Path("/om/user/imgriff/datasets/human_word_rec_SWC_2023/sounds/"),
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
    args = parser.parse_args()

    run_eval(args)


if __name__ == "__main__":
    cli_main()
