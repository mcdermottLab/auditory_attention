import pathlib
from argparse import ArgumentParser
import yaml
import pickle
import csv
import torch 
import soxr
import h5py
import numpy as np 
import pandas as pd
from tqdm.auto import tqdm
from pytorch_lightning import seed_everything
from src.spatial_attn_architecture import CueDurationCNN2DExtractor, CueDurationCNNNew 
from corpus.swc_mono_test import SWCMonoTestSet
import src.audio_transforms as at
import src.custom_modules as cm

seed_everything(1)

torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# torch module to center crop the cochleagram to a given duration 
class CenterCropCoch(torch.nn.Module):
    def __init__(self, crop_duration, sig_duration, sr, pad_dur=False):
        super().__init__()
        self.n_crop_frames = int(crop_duration * sr)
        sig_frames = int(sig_duration * sr)
        self.crop_context = sig_frames - self.n_crop_frames
        self.crop_start = self.crop_context // 2
        self.crop_end = self.crop_start + self.n_crop_frames
        self.pad_to_dur = False
        if pad_dur > crop_duration:
            print("Padding to duration")
            self.pad_to_dur = True 
            self.pad_context = int(pad_dur * sr) - self.n_crop_frames

    def forward(self, x):
        # crop x to n_crop_frames and zero pad back to original length
        cropped = x[..., self.crop_start:self.crop_end]
        if self.pad_to_dur:
            cropped = torch.nn.functional.pad(cropped, (0, self.pad_context), "constant", 0)
        return cropped 


def reformat_state_dict(checkpoint, ln_affine=True):
    new_state_dict = {}
    for k,v in  checkpoint['state_dict'].items():
        if ln_affine:
            # update key for easy norm layer access
            if 'conv' in k and '.0.' in k:
                k = k.replace('conv_block_', 'layer_norm_')
                k = k.replace('.0.', '.')
            # decrement conv layer ixs in dict to match model
            elif 'conv' in k and '.1.' in k:
                k = k.replace('.1.', '.0.')
            # decrement pool layer rixs in dict to match model
            elif 'conv' in k and '.3.' in k:
                k = k.replace('.3.', '.2.')
        new_state_dict[k] = v

    return new_state_dict

def run_eval(args):

    model_name = pathlib.Path(args.config).stem
    checkpoint_path = args.ckpt_path
    config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
    print(f"Loading model from {checkpoint_path}")
    
    label_type = 'CV'

    # set audio transforms
    sr = config['audio']['rep_kwargs']['sr']
    coch_sr = config['audio']['rep_kwargs']['env_sr']
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
    # set up test params 
    with open(args.test_manifest_path, 'rb') as file:
        test_manifest = pickle.load(file)

    dataset_cond_ix, cue_dur = test_manifest[args.array_id]

    n_cue_frames = int(cue_dur * coch_sr)
    config['model']['n_cue_frames'] = n_cue_frames
    signal_dur = config['audio']['rep_kwargs']['out_dur']
    config['cue_duration_test'] = True

    pad_dur = 0.5
    cue_crop_op = CenterCropCoch(cue_dur,
                                 signal_dur,
                                 coch_sr,
                                 pad_dur=pad_dur)

    # Load model
    ln_affine = config['model'].get('ln_affine', True)
    if 'no_affine' in args.config or not ln_affine:
        model = CueDurationCNNNew(**config['model'])
    else:
        model = CueDurationCNN2DExtractor(**config['model'])

    # reformat state dict to match model 
    checkpoint = torch.load(checkpoint_path)
    checkpoint['state_dict'] = {k.replace('model._orig_mod.', ''):v for k,v in checkpoint['state_dict'].items()}
    new_state_dict = reformat_state_dict(checkpoint, ln_affine=config['model'].get('ln_affine', True))
    norm_param_dict = {k:v for k,v in new_state_dict.items() if 'norm' in k}
    
    # populate layer norm params 
    for key, param in norm_param_dict.items():
        layer_name = key.split('.')[1]
        n_cue_frames_at_layer = model.layer_norm_params[f'{layer_name}']['ln_cue_frames']
        param_type = 'weight' if 'weight' in key else 'bias'
        model.layer_norm_params[f'{layer_name}'][f'{param_type}'] = param
        model.layer_norm_params[f'{layer_name}'][f'cue_{param_type}'] = param[..., : n_cue_frames_at_layer]
    
    # load checkpoint 
    model.load_state_dict(new_state_dict, strict=False) # strict=False to skip attn weights 
    model = model.eval().cuda()

    # load and freeze model
    coch_gram = cm.AttnAudioInputRepresentation(**config['audio']).cuda()

    dataset = SWCMonoTestSet(stim_path=args.stim_path,
                            cond_ix=dataset_cond_ix,
                            model_sr=sr,
                            label_type=label_type)
    
    condition, snr = dataset.stim_cond_map[dataset_cond_ix]
    print(f"Evaluating {model_name} on {condition} at {snr}db SNR with {cue_dur} cue duration.")

    def collate_fn(batch):
        #apply transforsms to batch
        cues = torch.stack([audio_transforms(cue, None)[0] for cue, _, _ in batch])
        mixtures = torch.stack([audio_transforms(mix, None)[0] for _, mix,  _ in batch])
        labels = torch.tensor([label for _, _, label in batch]).type(torch.LongTensor)
        return cues, mixtures, labels

    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=16,
                                             shuffle=False,
                                             collate_fn=collate_fn,
                                             num_workers=args.n_jobs)
    

    out_dir = args.exp_dir / model_name 
    # make dir if it doesn't exist
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{model_name}_{condition}_{snr}dB_SNR_{int(cue_dur * 1000)}ms_eval_results.csv", 'w') as file:
        csv_out = csv.writer(file, delimiter=",")
        # write column header to csv
        print("writing csv header")
        csv_out.writerow(['pred_word_int', 'true_word_int', 'accuracy'])
        
        # run eval 
        for i, batch in enumerate(tqdm(dataloader, desc=f"evaluating {model_name} on {condition} at {snr}dB SNR")):
            cue, mixture, word = batch
            # to device 
            cue = cue.cuda()
            mixture = mixture.cuda()

            cue, mixture = coch_gram(cue, mixture)

            # crop cue duration
            cue = cue_crop_op(cue)
            logits = model(cue, mixture, None)

            preds = logits.softmax(dim=-1).argmax(dim=-1).cpu().detach().numpy().astype('int')
            true_word = word.numpy().astype('int')
            accuracy = (true_word == preds).astype('int')
            # write to csv
            rows = list(zip(*[preds, true_word, accuracy]))
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
        "--stim_path",
        default=pathlib.Path("/om/user/imgriff/datasets/human_word_rec_SWC_2023/sounds/"),
        type=pathlib.Path,
        help="Path where background stimuli are saved",
    )
    parser.add_argument(
        "--test_manifest_path",
        default=pathlib.Path(""),
        type=pathlib.Path,
        help="Path map from array ixs to test conditions pickle file.",
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
