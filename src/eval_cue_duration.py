import pathlib
from argparse import ArgumentParser
import yaml
import pickle
import csv
import torch 
import soxr
import re 
import h5py
import numpy as np 
import pandas as pd
from tqdm.auto import tqdm
from pytorch_lightning import seed_everything
from src.spatial_attn_architecture import CueDurationCNNNew 
from corpus.swc_mono_test import SWCMonoTestSetH5Dataset
import src.audio_transforms as at
import src.custom_modules as cm


seed_everything(1)

torch.set_float32_matmul_precision('medium')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# torch module to center crop the cochleagram to a given duration 

class CenterCropCuePad(torch.nn.Module):
    """
    Center crops the cue to make a shorter signal.
    """
    def __init__(self, signal_size, crop_length, signal_sr, pad_dur=False):
        super(CenterCropCuePad, self).__init__()
        self.crop_length = int(crop_length * signal_sr)
        self.signal_size = int(signal_size * signal_sr)
        self.sr = signal_sr 
        self.start_crop_center = int((self.signal_size-self.crop_length)/2)

        # compute pad context for zero padding back to original length
        self.pad_context = (self.signal_size - self.crop_length) // 2
        self.pad_to_dur = False 
        if pad_dur > crop_length:
            print("Padding to duration")
            self.pad_to_dur = True 
            self.pad_context = (int(pad_dur * signal_sr) - self.crop_length) // 2 
        print(f"Pad context: {self.pad_context}")
        print(f"Pad to dur: {self.pad_to_dur}")
        print(f"Start crop center: {self.start_crop_center}")
        print(f"Signal size: {self.signal_size}")
        print(f"Crop length: {self.crop_length}")

    def forward(self, cue_wav):
        """
        Args:
            cue_wav (torch.Tensor): the waveform that will be used as
                the cue audio sample (usually speech)
        """
        cue_wav = cue_wav[..., self.start_crop_center : self.start_crop_center + self.crop_length]
        # pad cue back to original length
        if self.pad_to_dur:
            cue_wav = torch.nn.functional.pad(cue_wav, (self.pad_context, self.pad_context), "constant", 0)
            if cue_wav.shape[-1] % 2 != 0:
                cue_wav = torch.nn.functional.pad(cue_wav, (0, 1), "constant", 0)
        return cue_wav
    

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
            if 'coch' in k:
                continue
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

    # set up test params 
    with open(args.test_manifest_path, 'rb') as file:
        test_manifest = pickle.load(file)

    distractor_cond, cue_dur = test_manifest[args.array_id]

    n_cue_frames = int(cue_dur * coch_sr)
    config['model']['n_cue_frames'] = n_cue_frames
    signal_dur = config['audio']['rep_kwargs']['out_dur']
    signal_sr = config['audio']['rep_kwargs']['sr']

    config['cue_duration_test'] = True

    pad_dur = 0.5
    orig_audio_dur = 2.5
    cue_crop_op = CenterCropCuePad(orig_audio_dur, cue_dur, signal_sr, pad_dur=pad_dur).cuda() 

    model = CueDurationCNNNew(**config['model'])


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
        # use middle frames of layer norm params
        start = int((param.shape[-1] - n_cue_frames_at_layer) / 2)
        end = start + n_cue_frames_at_layer
        model.layer_norm_params[f'{layer_name}'][f'cue_{param_type}'] = param[..., start : end]
        
    
    # load checkpoint 
    model.load_state_dict(new_state_dict, strict=False) # strict=False to skip coch & LN weights 
    model = model.eval().cuda()

    # load and freeze model

    diotic_transforms = at.AudioCompose([
                    at.AudioToTensor(),
                    at.CombineWithRandomDBSNR(low_snr=0, high_snr=0), 
                    at.RMSNormalizeForegroundAndBackground(rms_level=0.02),  # 0.02 is the default for CV-based models 
                    at.DuplicateChannel(),

            ])
    diotic_transforms = diotic_transforms.cuda()


    coch_gram = cm.AttnAudioInputRepresentation(**config['audio']).cuda()

    dataset = SWCMonoTestSetH5Dataset(h5_path="/om/user/imgriff/datasets/human_word_rec_SWC_2024/model_eval_stim.h5",
                                            eval_distractor_cond=distractor_cond,
                                            model_sr=44100,
                                            label_type='CV')
    
    print(f"Evaluating {model_name} on {distractor_cond} at 0db SNR with {cue_dur} cue duration.")


    def single_signal_collate_fn(batch):
        #apply transforsms to batch
        cues = torch.stack([diotic_transforms(cue, None)[0] for cue, fg, bg, label, confusion in batch])
        mixtures = torch.stack([diotic_transforms(fg, bg)[0] for cue, fg, bg, label, confusion in batch]).type(torch.FloatTensor)
        labels = torch.tensor([label for cue, fg, bg, label, confusion in batch]).type(torch.LongTensor)
        confusion = torch.tensor([confusion for cue, fg, bg, label, confusion in batch]).type(torch.LongTensor)
        return cues, mixtures, labels, confusion

    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=16,
                                             shuffle=False,
                                             collate_fn=single_signal_collate_fn,
                                             num_workers=args.n_jobs)
    

    out_dir = args.exp_dir / model_name 
    # make dir if it doesn't exist
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{model_name}_{distractor_cond}_0dB_SNR_{int(cue_dur * 1000)}ms_eval_results.csv", 'w') as file:
        csv_out = csv.writer(file, delimiter=",")
        # write column header to csv
        print("writing csv header")
        csv_out.writerow(['pred_word_int', 'true_word_int', 'accuracy'])
        
        # run eval 
        for i, batch in enumerate(tqdm(dataloader, desc=f"evaluating {model_name} on {distractor_cond} at 0dB SNR")):
            cue, mixture, word, _ = batch
            # to device 
            cue = cue.cuda()
            # crop cue duration
            cue = cue_crop_op(cue)
            mixture = mixture.cuda()
            cue, mixture = coch_gram(cue, mixture)

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
