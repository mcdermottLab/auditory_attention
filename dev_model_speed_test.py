import torch 
import numpy as np 
import h5py
import os
from pathlib import Path
# import IPython.display as ipd

import src.spatial_attn_lightning as binaural_lightning 
import yaml
from pytorch_lightning import Trainer

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

torch.set_float32_matmul_precision('medium')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# torch._dynamo.config.verbose=True

config_path = "config/binaural_attn/word_task_mixed_cue_v04_80p_co_located_torch_2.yml"
config = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)

config['num_workers'] = 4
config['hparas']['batch_size'] = 64
config['audio']['rep_kwargs']['rep_on_gpu'] = True


model = binaural_lightning.BinauralAttentionModule(config)
# model = torch.compile(model)
trainer = Trainer(
    precision="32",
    limit_val_batches=0.0,
    num_nodes=1,
    benchmark=True,
    devices=1, # was gpus=1,
    # detect_anomaly=True,
    # strategy="ddp",
    accelerator="gpu",
)
trainer.fit(model)



