import sys
import os
import yaml
import h5py 
import torch
from pathlib import Path
import numpy as np 
from argparse import ArgumentParser
import src.audio_transforms as at
from src.spatial_attn_lightning import BinauralAttentionModule 
import pandas as pd 
from tqdm.auto import tqdm
import pickle
import soxr

from corpus.binaural_attention_h5 import BinauralAttentionDataset

torch.set_float32_matmul_precision('medium')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch._dynamo.config.skip_nnmodule_hook_guards=False


def save_activations(f, layer, activations, row, n_rows_to_save, time_average=False):
    """Save activations to the HDF5 file."""
    if time_average and 'relufc' not in layer:
        activations = activations.mean(dim=-1, keepdim=True)
    
    # Use layer name directly as dataset name (no suffix)
    dataset_name = layer
    if dataset_name not in f:
        f.create_dataset(dataset_name, shape=[n_rows_to_save, np.prod(activations.shape)], dtype=np.float32)
    f[dataset_name][row] = activations.cpu().view(-1).numpy()


def get_activations(args):
    # set random seeds 
    torch.manual_seed(0)
    np.random.seed(0)
  
    # Get config for model
    checkpoint_path = args.ckpt_path
    
    if args.config != "":
        config_path = Path(args.config)
    else:
        with open(args.config_list, 'rb') as f:
            model_config = pickle.load(f)
        config_output = model_config[args.job_id]
        if isinstance(config_output, dict):
            print(f"Loading config from {config_output['config_path']}")
            config_path, checkpoint_path = config_output['config_path'], config_output['ckpt_path']
            config_path = Path(config_path)
        else:
            config_path = Path(config_output)
            
    print(config_path)
    config = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)
    config['corpus']['cue_free_percentage'] = 0.0
    model_name = config_path.stem

    # Set audio transforms  
    snr = 0 
    audio_transforms = at.AudioCompose([
                    at.AudioToTensor(), 
                    at.BinauralRMSNormalizeForegroundAndBackground(rms_level=0.02,
                                                                   v2_demean=True),
            ])
    audio_transforms = audio_transforms.cuda()
    

    # handle checkpoint path
    if checkpoint_path == "":
        ckpt_dir = Path('attn_cue_models/') / model_name / 'checkpoints'
        checkpoint_path = sorted(ckpt_dir.glob("*.ckpt"), key=os.path.getctime)[-1]

    strict_ckpt = True 
    if 'backbone' in model_name:
        config['model']['backbone_with_ecdf_gains'] = True
        strict_ckpt = False
        model_name = f"{model_name}_with_ecdf_gains"

    print(f"Loading {model_name} from {checkpoint_path}")
    rand_weight_str = ""
    if not args.random_weights:
        model = BinauralAttentionModule.load_from_checkpoint(checkpoint_path=checkpoint_path,
                                                             config=config,
                                                             strict=strict_ckpt).eval().cuda()
    else:
        model = BinauralAttentionModule(config=config).eval().cuda()
        rand_weight_str = "_rand_weights"
    coch_gram = model.coch_gram.cuda()
    label_type = 'CV'
    sr = config['audio']['rep_kwargs']['sr']
          
    # get dataset
    dataset = BinauralAttentionDataset(**config['corpus'], batch_size=1, mode=args.data_split,)
    
    dataloader = torch.utils.data.DataLoader(
                            dataset,
                            batch_size=1,
                            shuffle=False,
                            num_workers=args.n_jobs)

    ########################
    # Set hooks for backbone
    ########################
    activations = {}
    def get_activation(name):
        def hook(model, input, output):
            if name in activations:
                activations[name] = torch.cat((activations[name], output.detach()), dim=0)
            else:
                activations[name] = output.detach()
        return hook

    if hasattr(model.model, '_orig_mod'):
        modules = {name:module for name, module in model.model._orig_mod.model_dict.named_children()}
        relu_fc = model.model._orig_mod.relufc
    else:
        modules = {name:module for name, module in model.model.model_dict.named_children()}
        relu_fc = model.model.relufc

    modules = {**modules, **{'relufc': relu_fc}}
    for name, module in modules.items():
        if 'conv' in name:
            module[0].register_forward_hook(get_activation(f"{name}_ln"))
            module[2].register_forward_hook(get_activation(f"{name}_relu"))
        if 'ecdf' in model_name and 'attn' in name:
            continue
        else:
            module.register_forward_hook(get_activation(name))

    model = model.eval().cuda()
    outname = Path(f'binaural_unit_activations_for_SVM/{model_name}{rand_weight_str}/{model_name}{rand_weight_str}_model_activations_for_word_SVM_{args.data_split}.h5')
    out_dir = Path("/om/scratch/Fri/imgriff")
    outname = out_dir / outname 

    layer_shape_dict_name = Path(f'binaural_unit_activations_for_SVM/{model_name}/{model_name}_layer_shape_dict.pkl')
    layer_shape_dict_name.parent.mkdir(parents=True, exist_ok=True)
    outname.parent.mkdir(parents=True, exist_ok=True)
    print(f"Preparing to write activations to {outname}")
    
    n_activations = args.n_activations
    n_words = args.n_words
    layer_shape_dict = {}

    # Track word counts
    word_class_counts = {word_ix: 0 for word_ix in range(n_words)}

    if outname.exists() and not args.overwrite:
        if args.resume_progress:
            print(f"{outname} exists. Resuming progress...")
            # Count existing examples per word
            with h5py.File(outname, 'a') as f:
                if 'target_word_int' in f:
                    existing_labels = f['target_word_int'][:]
                    for label in existing_labels:
                        if label != 0 and 0 <= label < n_words:
                            word_class_counts[int(label)] += 1
            print(f"Current word counts: min={min(word_class_counts.values())}, max={max(word_class_counts.values())}")
        else:
            print(f"{outname} already exists. Exiting.")
            sys.exit()

    n_rows_to_save = n_words * n_activations
    
    with h5py.File(outname, 'a') as f:
        # Initialize target_word_int dataset if it doesn't exist
        if 'target_word_int' not in f:
            f.create_dataset('target_word_int', shape=[n_rows_to_save], dtype=np.int32)
        
        with torch.no_grad():
            dataloader_iter = iter(dataloader)
            total_processed = 0
            
            pbar = tqdm(total=n_words * n_activations, desc='Processing activations')
            
            while min(word_class_counts.values()) < n_activations:
                try:
                    batch = next(dataloader_iter)
                except StopIteration:
                    # Reset dataloader if we run out of data
                    dataloader_iter = iter(dataloader)
                    batch = next(dataloader_iter)
                
                # Get signals 
                cue, target, background, label = batch
                target = target.squeeze(0)
                word_ix = int(label.item())
                
                # Skip if this word already has enough examples
                if word_ix >= n_words or word_class_counts[word_ix] >= n_activations:
                    continue
                
                # Calculate global row index for this word
                row = word_ix * n_activations + word_class_counts[word_ix]
                
                # Check if already processed
                if args.resume_progress and 'cochleagram' in f and f['cochleagram'][row].sum() != 0:
                    word_class_counts[word_ix] += 1
                    pbar.update(1)
                    continue
                
                # Process audio
                target, _ = audio_transforms(target, None)
                _, target = coch_gram(_, target.cuda())
                
                # Save label
                f['target_word_int'][row] = word_ix
                
                # Save cochleagram
                save_activations(f, 'cochleagram', target, row, n_rows_to_save, 
                               time_average=args.time_average)
                
                # Get model activations
                activations = {}
                model(None, target, None)
                
                for layer, acts in activations.items():
                    if 'relu' not in layer:
                        continue # only save relu layers to save space
                    if len(acts) == 2:
                        _, acts = acts
                    save_activations(f, layer, acts, row, n_rows_to_save, 
                                   time_average=args.time_average)
                
                # Save layer shapes on first iteration
                if total_processed == 0:
                    layer_shape_dict = {layer: activations[layer].shape for layer in activations.keys()}
                    shape_dict = {**layer_shape_dict}
                    with open(layer_shape_dict_name, 'wb') as p:
                        pickle.dump(shape_dict, p)
                    layer_names = [name.encode('utf-8') for name in activations.keys()]
                    if 'layer_names' not in f:
                        f.create_dataset('layer_names', data=layer_names)
                
                # Update counts
                word_class_counts[word_ix] += 1
                total_processed += 1
                pbar.update(1)
                
                # Clear memory
                activations = {}
            
            pbar.close()
    
    print(f"\nFinal word counts:")
    print(f"Min: {min(word_class_counts.values())}, Max: {max(word_class_counts.values())}")
    print(f"All words complete: {all(count == n_activations for count in word_class_counts.values())}")


def cli_main():
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default="", help='Path to experiment config.')
    parser.add_argument("--output_dir", default=Path("./binaural_unit_tuning"), type=Path)
    parser.add_argument("--ckpt_path", default=Path("./exp"), type=Path)
    parser.add_argument("--n_activations", default=100, type=int, 
                       help="Number of examples per word class")
    parser.add_argument("--n_jobs", default=0, type=int)
    parser.add_argument("--config_list", type=str)
    parser.add_argument("--job_id", default=0, type=int)
    parser.add_argument("--n_words", default=100, type=int)
    parser.add_argument("--random_weights", action='store_true')
    parser.add_argument("--cue_single_source", action='store_true')
    parser.add_argument("--overwrite", action='store_true')
    parser.add_argument("--resume_progress", action='store_true')
    parser.add_argument("--data_split", type=str, default="train", 
                       help="Dataset split to use")
    parser.add_argument("--time_average", action='store_true')
    args = parser.parse_args()

    get_activations(args)


if __name__ == "__main__":
    cli_main()