import torch 
import numpy as np 
import h5py
import os
from corpus.speaker_room_dataset import SpeakerRoomDataset
import src.audio_transforms as at
import src.spatial_attn_lightning as binaural_lightning 
import yaml
from tqdm.auto import tqdm
from pathlib import Path
import pandas as pd 
import soxr
from argparse import ArgumentParser


os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

torch.set_float32_matmul_precision('medium')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def get_brir(azim=None, elev=None, coords=None, h5_fn=None, IR_df=None, out_sr=44_100):
    if coords is not None:
        azim, elev = coords
    df_row = IR_df[(IR_df['src_azim'] == azim) & (IR_df['src_elev'] == elev)]
    brir_ix = df_row['index_brir'].values[0]
    sr_src = df_row['sr'].values[0]
    with h5py.File(h5_fn, 'r') as f:
        brir = f['brir'][brir_ix]
    if out_sr != sr_src:
        brir = soxr.resample(brir.astype(np.float32), sr_src, out_sr)
    return brir


def run_search(args):
    '''
    Run gridsearch over spatial configurations to find maximally distracting configuration
    for a target and distractor distractor signal. 
    Finds azimuth of distractor that that maximizes word classification loss.
    '''
    # set np and torch random seeds 
    np.random.seed(args.job_ix)
    torch.manual_seed(args.job_ix)

    # get config 
    config_path = Path(args.config_path)
    config = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)
    config['hparas']['batch_size'] = 1 # config['data']['loader']['batch_size'] // args.gpus
    config['num_workers'] = 4
    config['noise_kwargs']['low_snr'] = 0
    config['noise_kwargs']['high_snr'] = 0
    # get model input sr for brir resampling
    signal_sr = config['audio']['rep_kwargs']['sr']
    coch_sr = config['audio']['rep_kwargs']['env_sr']

    # get trained model 
    ckpt_path = Path(args.checkpoint_path)
    model = binaural_lightning.BinauralAttentionModule.load_from_checkpoint(checkpoint_path=ckpt_path,
                                                                            config=config).cuda().eval()
    coch_transform = model.coch_gram.cuda()

    # init dataset 
    dataset = SpeakerRoomDataset(manifest_path='/om2/user/rphess/Auditory-Attention/final_binaural_manifest.pkl',
                            excerpt_path='/om2/user/msaddler/spatial_audio_pipeline/assets/swc/manifest_all_words.pdpkl',
                            cue_type='voice_and_location',
                            sr=signal_sr) 
    batch_size = 20 
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=config['num_workers'])

    # init audio transforms 
    audio_transforms_0_db = at.AudioCompose([
                    at.AudioToTensor(),
                    at.BinauralCombineWithRandomDBSNR(low_snr=0,    # is 0 dB
                                                    high_snr=0), # is 0 dB 
                    at.BinauralRMSNormalizeForegroundAndBackground(rms_level=0.02), # 20 * np.log10(0.02/20e-6) = 60 dB SPL 
            ])
    audio_transforms_0_db = audio_transforms_0_db.cuda()

    # init brir search
    test_IR_manifest_dir = Path(args.room_manifest_path) #  Path("/om2/user/imgriff/spatial_audio_pipeline/assets/brir/mit_bldg46room1004_min_reverb")
    room_ix = args.room_ix
    test_IR_manifest_path = test_IR_manifest_dir / "manifest_brir.pdpkl"
    h5_fn = test_IR_manifest_dir / f"room{room_ix:04}.hdf5"
    new_room_manifest = pd.read_pickle(test_IR_manifest_path)
    only14_manifest = new_room_manifest[(new_room_manifest['index_room'] == room_ix)  & (new_room_manifest['src_dist'] == 1.4)]

    #### init search parameters
    n_examples = args.n_examples

    target_azim, target_elev = 0, 0
    distractor_elev = 0 

    target_brir = get_brir(azim=target_azim, elev=target_elev, h5_fn=h5_fn, IR_df=only14_manifest, out_sr=signal_sr)
    sp_to_target_loc = at.Spatialize(target_brir, model_sr=signal_sr).cuda()

    azims_to_search = only14_manifest.src_azim.unique()
    loss_fn = torch.nn.CrossEntropyLoss(reduction='none')

    # init output file:
    output_path = Path(args.output_path)
    output_path = output_path / config_path.stem
    output_path.mkdir(parents=True, exist_ok=True)
    output_file = output_path / f"{config_path.stem}_azim_search_seed{args.job_ix}.h5"

    with h5py.File(output_file, 'w') as f:
        f.create_dataset('losses', shape = [n_examples, len(azims_to_search)], dtype=np.float32)
        f.create_dataset('azimuths', data=azims_to_search, dtype='int')

        for azim_ix, azim in enumerate(tqdm(azims_to_search, desc=f"Searching over {len(azims_to_search)} azimuths...")):
            brir = get_brir(azim=azim, elev=distractor_elev, h5_fn=h5_fn, IR_df=only14_manifest, out_sr=signal_sr)
            sp_to_distractor_loc = at.Spatialize(brir, model_sr=signal_sr).cuda()
            
            for j, batch in enumerate(tqdm(dataloader, total=n_examples//batch_size, leave=False, desc=f"Batches: ")):
                # get batch for adding to losses 
                start = j * batch_size
                end = (j+1) * batch_size
                # spatialize cue and fg 
                cue, fg, bg, label, confusion = batch

                cue = sp_to_target_loc(cue.cuda())
                fg = sp_to_target_loc(fg.cuda())
                bg = sp_to_distractor_loc(bg.cuda()) 
                label = label.cuda()

                cue, _ = audio_transforms_0_db(cue, None)

                mixture_wav, _ = audio_transforms_0_db(fg, bg)
                cue, mixture = coch_transform(cue, mixture_wav)
                
                with torch.no_grad():
                    pred = model(cue, mixture, None)
                    loss = loss_fn(pred, label)
                f['losses'][start:end, azim_ix] = loss.cpu().numpy()
                
                if (j+1) * batch_size >= n_examples:
                    break

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str,  help="Path to model checkpoint")
    parser.add_argument("--config_path", type=str, help="Path to model config file")
    parser.add_argument("--output_path", type=str,  help="Path to save signals")
    parser.add_argument("--job_ix", type=int,  default=0, help="Slurm job array ix - used to init random seed")
    parser.add_argument("--n_examples", type=int, default=500, help="Number of examples to run")
    parser.add_argument("--room_manifest_path", 
                        default="/om2/user/imgriff/spatial_audio_pipeline/assets/brir/mit_bldg46room1004_min_reverb",
                        type=str, help="Path to room manifest")
    parser.add_argument("--room_ix", type=int, default=0, help="Room index code to use for BRIRs")
    args = parser.parse_args()
    run_search(args)



