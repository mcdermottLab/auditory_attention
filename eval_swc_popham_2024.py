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
from src.spatial_attn_lightning import BinauralAttentionModule 
from corpus.swc_popham_test_h5 import SWCPophamCondTestSet2024
import src.audio_transforms as at

seed_everything(1)

torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
    
def run_eval(args):

    model_name = pathlib.Path(args.config).stem
    checkpoint_path = args.ckpt_path
    config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
    print(f"Loading model from {checkpoint_path}")
    
    # load model 
    module = BinauralAttentionModule
    label_type = 'CV'

    # set audio transforms
    sr = config['audio']['rep_kwargs']['sr']
    audio_config = config['audio']

    # get snr for audio transforms if part of the config
    with open(args.stim_cond_map, 'rb') as f:
        condition_dict = pickle.load(f)
    test_cond = condition_dict[args.array_id]
    target_harmonicity = test_cond['target_harmonicity']
    distractor_harmonicity = test_cond['distractor_harmonicity']
    
    snr = 0
    condition = f"{target_harmonicity}_target_{distractor_harmonicity}_distractor"
    dataset = SWCPophamCondTestSet2024(root=args.stim_path,
                            target_harmonicity=target_harmonicity,
                            distractor_harmonicity=distractor_harmonicity)

    audio_transforms = at.AudioCompose([
                at.AudioToTensor(),
                at.CombineWithRandomDBSNR(low_snr=snr, high_snr=snr), 
                at.RMSNormalizeForegroundAndBackground(rms_level=0.02),  # 0.02 is the default for CV-based models 
                at.UnsqueezeAudio(dim=0),
                at.DuplicateChannel()
                ])

    # load and freeze model
    model = module.load_from_checkpoint(checkpoint_path=checkpoint_path, config=config).eval().cuda()

    coch_gram = model.coch_gram.cuda()
    

    print(f"Evaluating {model_name} on {condition} at {snr}db SNR")
    def collate_fn(batch):
        cues, mixtures, labels, sex_cond_int, orig_df_row_ix = [], [], [], [], []
        
        for cue, target, bg, label, sex_cond, orig_ix in batch:
            cues.append(audio_transforms(cue, None)[0])
            mixtures.append(audio_transforms(target, bg)[0])
            labels.append(label)
            sex_cond_int.append(sex_cond)
            orig_df_row_ix.append(orig_ix)
        
        cues = torch.stack(cues)
        mixtures = torch.stack(mixtures)
        labels = torch.tensor(labels, dtype=torch.long)
        sex_cond_int = torch.tensor(sex_cond_int, dtype=torch.long)
        orig_df_row_ix = torch.tensor(orig_df_row_ix, dtype=torch.long)
        
        return cues, mixtures, labels, sex_cond_int, orig_df_row_ix


    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=12,
                                             shuffle=False,
                                             collate_fn=collate_fn,
                                             num_workers=args.n_jobs)

    # set up output file 
    out_dir = args.exp_dir / model_name 
    # make dir if it doesn't exist
    out_dir.mkdir(parents=True, exist_ok=True)
    # track running average of accuracy and confusions 
    acc_sum = 0
    with open(out_dir / f"{model_name}_{condition}_{snr}dB_SNR_eval_results.csv", 'w') as file:
        csv_out = csv.writer(file, delimiter=",")
        # write column header to csv
        print("writing csv header")
        csv_out.writerow(['pred_word_int', 'true_word_int', 'accuracy', 'sex_cond_int', 'orig_df_row_ix'])

        # run eval 
        for i, batch in enumerate(tqdm(dataloader, desc=f"evaluating {model_name} on {condition} at {snr}dB SNR")):
            cue, mixture, word, sex_cond_int, orig_df_row_ix = batch
            

            # to device 
            cue = cue.cuda()
            mixture = mixture.cuda()

            cue, mixture = coch_gram(cue, mixture)
            logits = model(cue, mixture, None)

            preds = logits.softmax(dim=-1).argmax(dim=-1).cpu().detach().numpy().astype('int')
            # handle data types
            true_word = word.numpy().astype('int')
            sex_cond_int = sex_cond_int.numpy().astype('int')
            orig_df_row_ix = orig_df_row_ix.numpy().astype('int')
            # calculate accuracy
            accuracy = (true_word == preds).astype('int')
            acc_sum += accuracy.sum()
            # write to csv
            rows = list(zip(*[preds, true_word, accuracy, sex_cond_int, orig_df_row_ix]))
            csv_out.writerows(rows)
            if i == 0:
                print(f"EG of data writing: {rows}")
            if i % 100 == 0:
                print(f"writing on batch {i} of {len(dataloader)}")
                file.flush() # only write every 100 batches 
        # print final accuracy
        acc = acc_sum / len(dataset)
        print(f"Final accuracy: {acc}")
        
def cli_main():
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default="", help='Path to experiment config.')
    parser.add_argument(
        "--exp_dir",
        default=pathlib.Path("./exp"),
        type=pathlib.Path,
        help="Directory to save test results in. (Default: './exp')",
    )
    parser.add_argument(
        "--stim_path",
        default=pathlib.Path("/om/user/imgriff/datasets/human_swc_popham_exmpt_2024/model_eval_h5s/"),
        type=pathlib.Path,
        help="Path where background stimuli are saved",
    )
    parser.add_argument(
        "--stim_cond_map",
        default="all_stim_swc_popham_exmpt_2024_cond_manifest.pkl",
        help="Path to pickle file containing condition map for stimuli",
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
