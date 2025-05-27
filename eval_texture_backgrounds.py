import re

import yaml
import csv
import torch 
import numpy as np 
import pandas as pd
from tqdm.auto import tqdm
import pathlib
from pathlib import Path 
from argparse import ArgumentParser
from pytorch_lightning import seed_everything
from src.spatial_attn_lightning import BinauralAttentionModule 
from corpus.speech_and_texture_test import SpeechAndTextureTestSet
import src.audio_transforms as at

seed_everything(1)

torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# Set test directory dictionary 
stim_path_dict = {
    0: '/om/user/imgriff/datasets/speech_in_synthetic_textures/snrN03/stim.hdf5',
    1: '/om/user/imgriff/datasets/speech_in_synthetic_textures/snrN06/stim.hdf5',
    2: '/om/user/imgriff/datasets/speech_in_synthetic_textures/snrN09/stim.hdf5',
}


def run_eval(args):
    model_name = pathlib.Path(args.config).stem
    checkpoint_path = args.ckpt_path
    config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
    print(f"Loading model from {checkpoint_path}")
    
    # load model 
    if 'binaural_attn' in args.config:
        module = BinauralAttentionModule
        label_type = 'CV'
    else:
        module = AttentionalTrackingModule
        config['data']['audio']['rep_kwargs']['center_crop'] = True
        config['data']['audio']['rep_kwargs']['out_dur'] = 2
        label_type = "WSN"

    # set audio transforms
    sr = config['audio']['rep_kwargs']['sr']
    audio_config = config['audio']
    IIR_COCH = not audio_config['rep_kwargs']['rep_on_gpu']

    if IIR_COCH:
        audio_transforms = at.AudioCompose([
                at.AudioToTensor(),
                # at.CombineWithRandomDBSNR(low_snr=snr, high_snr=snr), 
                at.RMSNormalizeForegroundAndBackground(rms_level=0.1),  # 0.1 is the default for the swc-based models 
                at.UnsqueezeAudio(dim=0),
                at.AudioToAudioRepresentation(**audio_config),
            ])
    if 'mono' not in args.config:
        print(f"Using diotic input")
        audio_transforms = at.AudioCompose([
                    at.AudioToTensor(),
                    # at.CombineWithRandomDBSNR(low_snr=snr, high_snr=snr), 
                    at.RMSNormalizeForegroundAndBackground(rms_level=0.02),  # 0.02 is the default for CV-based models 
                    at.UnsqueezeAudio(dim=0),
                    at.DuplicateChannel()
            ])  
    else:
        audio_transforms = at.AudioCompose([
                    at.AudioToTensor(),
                    # at.CombineWithRandomDBSNR(low_snr=snr, high_snr=snr), 
                    at.RMSNormalizeForegroundAndBackground(rms_level=0.02),  # 0.02 is the default for CV-based models 
                    at.UnsqueezeAudio(dim=0),
            ])  

    # load and freeze model
    model = module.load_from_checkpoint(checkpoint_path=checkpoint_path, config=config).eval().cuda()
    coch_gram = None
    if 'v0' in args.config or 'v1' in args.config:
        coch_gram = model.coch_gram.cuda()


    stim_path = Path(stim_path_dict[args.array_id])
    dataset = SpeechAndTextureTestSet(file_path=stim_path,
                                      label_type=label_type)
    
    # get snr from stim path name 
    snr = int(re.findall(r'\d+', stim_path.parent.stem)[0])
    snr = -snr if 'N' in stim_path.parent.stem else snr

    print(f"Evaluating {model_name} on textures at {snr} dB SNR")

    def collate_fn(batch):
        #apply transforsms to batch
        cues = torch.stack([audio_transforms(cue, None)[0] for cue, _, _, _ in batch])
        mixtures = torch.stack([audio_transforms(mix, None)[0] for _, mix,  _, _ in batch])
        labels = torch.tensor([label for _, _, label, _ in batch]).type(torch.LongTensor)
        texture_ints = torch.tensor([texture for _, _, _, texture in batch]).type(torch.LongTensor)
        return cues, mixtures, labels, texture_ints

    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=10, # set batch size in dataset
                                             shuffle=False,
                                             collate_fn=collate_fn,
                                             num_workers=args.n_jobs)
    
    # set up output file 
    out_dir = args.exp_dir / model_name 
    # make dir if it doesn't exist
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{model_name}_{snr}dB_SNR_eval_results.csv", 'w') as file:
        csv_out = csv.writer(file, delimiter=",")
        # write column header to csv
        print("writing csv header")
        csv_out.writerow(['pred_word_int', 'true_word_int', 'accuracy', 'texture_int'])
        
        # run eval 
        for i, batch in enumerate(tqdm(dataloader, desc=f"evaluating {model_name} on textures at {snr}dB SNR")):
            cue, mixture, word, textures = batch
            # to device 
            cue = cue.cuda()
            mixture = mixture.cuda()

            # if spatialize:
            #     cue = spatialize(cue)
            #     mixture = spatialize(mixture)

            if coch_gram: # if cochleagram is not part of model arch. 
                cue, mixture = coch_gram(cue, mixture)

            if module == BinauralAttentionModule:
                logits = model(cue, mixture, None)
            else:
                logits = model(cue, mixture)
            preds = logits.softmax(dim=-1).argmax(dim=-1).cpu().detach().numpy().astype('int')
            true_word = word.numpy().astype('int')
            accuracy = (true_word == preds).astype('int')
            # write to csv
            rows = list(zip(*[preds, true_word, accuracy, textures.numpy()]))
            csv_out.writerows(rows)
            if i == 0:
                print(f"EG of data writing: {rows}")
            if i % 100 == 0:
                print(f"writing on batch {i} of {len(dataloader)}")
                file.flush() # only write every 100 batches 

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
