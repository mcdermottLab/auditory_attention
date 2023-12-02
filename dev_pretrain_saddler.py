
import torch 
import numpy as np 
import os
from pathlib import Path
import src.util_tfrecord as util_tfrecords
import os
import tensorflow as tf
import yaml
from pytorch_lightning import Trainer

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

torch.set_float32_matmul_precision('medium')

from collections import namedtuple
from typing import List, Tuple, Optional, Union
import numpy as np
import torch
from torchmetrics.classification import Accuracy
from pytorch_lightning import LightningModule

import src.audio_transforms as at
import src.custom_modules as cm
from src.base_word_loc_cnn import AuditoryCNN

class BinauralAttentionModule(LightningModule):
    def __init__(
        self,
        config: dict,
    ):
        super().__init__()
        # Make config sent to init
        self.config = config
        self.audio_config = config['audio']
        self.corpora_config = config['corpus']
        self.model_config = config['model']
        self.hparas_config = config['hparas']

        self.corpora_name = config.get('corpora_name', False)

        num_words = 801
        num_locs = 505

        self.audio_transforms = at.AudioCompose([
            at.AudioToTensor(),
            # at.BinauralCombineWithRandomDBSNR(low_snr=config['noise_kwargs']['low_snr'], high_snr=config['noise_kwargs']['high_snr']),
            at.BinauralRMSNormalizeForegroundAndBackground(rms_level=0.02), # 20 * np.log10(0.02/20e-6) = 60 dB SPL 
        ])

        # Init Model
        # fc_attn_only = self.model_config.get('fc_attn_only', False) 
        # fc_size = self.model_config.get('fc_size', 4096)
        # global_avg_cue = self.model_config.get('global_avg_cue', False)
        # Get model architecture
        model = AuditoryCNN(num_words=num_words, num_locs=num_locs) 
        # check if torch version 2 or greater - if so, compile model
        self.model = torch.compile(model, mode="reduce-overhead")

        # Add input rep to model or audio transforms
        self.rep_on_gpu = self.audio_config['rep_kwargs']['rep_on_gpu']
        self.coch_gram = cm.AttnAudioInputRepresentation(**self.audio_config).to(self.device)

        # Losses
        self.word_loss_fn = torch.nn.CrossEntropyLoss(ignore_index=0)
        self.loc_loss_fn = torch.nn.CrossEntropyLoss(ignore_index=0)

        self.train_acc = torch.nn.ModuleDict({'word':Accuracy(task="multiclass", num_classes=num_words), 
                                                'location':Accuracy(task="multiclass", num_classes=num_locs)})
        self.valid_acc = torch.nn.ModuleDict({'word':Accuracy(task="multiclass", num_classes=num_words),
                                                'location':Accuracy(task="multiclass", num_classes=num_locs)})
        self.test_acc = torch.nn.ModuleDict({'word':Accuracy(task="multiclass", num_classes=num_words),
                                                'location':Accuracy(task="multiclass", num_classes=num_locs)})
        self.test_confusion = torch.nn.ModuleDict({'word': Accuracy(task="multiclass", num_classes=num_words),
                                                 'location': Accuracy(task="multiclass", num_classes=num_locs)})
        self.accuracy = {'train': self.train_acc,
                         'val': self.valid_acc,
                         'test': self.test_acc,
                         'test_confusion': self.test_confusion
                        }

        # Constraints
        # self.attn_modules = [mod for name, mod in self.model._modules.items() if 'attn' in name]
        # self.bias_constraint = AttnBiasConstraint(min_val=0, max_val=1)
        # self.constrain_slope = self.model_config['attn_constraints'].get('slope', False)
        # if self.constrain_slope:
        #     self.slope_constraint = AttnSlopeConstraint(min_val=0)

        # Optimizer
        opt = getattr(torch.optim, self.hparas_config['optimizer'])
        model_params = [{'params': self.model.parameters()}]
        self.optimizer = opt(model_params, lr=self.hparas_config['lr'], eps=self.hparas_config['eps'])    

    def azim_elev_to_label(self, azim, elev):
        """
        """
        azim = azim.float()
        elev = elev.float()
        label = ((elev / 10) * 72) + (azim / 5) + 1
        return label.long()

    def _step(self, batch, batch_idx, step_type):
        if batch is None:
            return None

        # cue_features, cue_mask_ixs, scene_features, labels = batch

        cue_features = torch.from_numpy(batch['signal'].numpy())
        exs, time, channels = cue_features.shape
        cue_features = cue_features.view(exs, channels, 1, time).to(self.device)

        # process labels 
        word_label = torch.from_numpy(batch['word_int'].numpy()).type(torch.LongTensor).squeeze().to(self.device)
        loc_azim = torch.from_numpy(batch['label_loc_foreground_azim'].numpy())
        loc_elev = torch.from_numpy(batch['label_loc_foreground_elev'].numpy())
        location_label = self.azim_elev_to_label(loc_azim, loc_elev).squeeze().to(self.device)

        # features to cochleagram
        cue_features, _ = self.coch_gram(cue_features, None)
        # self() is self.forward()
        outputs = self(cue_features)

        word, location = outputs

        # filter valid examples for task losses
        w_idx = torch.argwhere(word_label > 0).squeeze()
        l_idx = torch.argwhere(location_label > 0).squeeze()
        # word screen
        word = word.index_select(0, w_idx)
        word_label = word_label.index_select(0, w_idx)
        # location screen
        location = location.index_select(0, l_idx)
        location_label = location_label.index_select(0, l_idx)

        # calc losses
        if w_idx.numel() > 0:
            word_loss = self.word_loss_fn(word, word_label)
            self.accuracy[step_type]['word'](word, word_label) # word accuracy
            self.log(f"{step_type}_word_acc", self.accuracy[step_type]['word'], on_step=False, on_epoch=True, prog_bar=True)
        else:
            word_loss = torch.tensor(0, device=self.device, dtype=torch.float32) #  

        if l_idx.numel() > 0:
            loc_loss = self.loc_loss_fn(location, location_label)
            self.accuracy[step_type]['location'](location, location_label) # location accuracy
            self.log(f"{step_type}_location_acc", self.accuracy[step_type]['location'], on_step=False, on_epoch=True, prog_bar=True)

        else:
            loc_loss = torch.tensor(0, device=self.device, type=torch.float32) #  
            
        loss = word_loss + loc_loss
        self.log(f"{step_type}_loss", loss.detach(), on_step=True, on_epoch=False, prog_bar=True)

        return loss

    def configure_optimizers(self):
        return [self.optimizer]

    def forward(self, sound: torch.tensor):
        outputs = self.model(sound)
        # Outputs here are logits
        return outputs

    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "val")

    def train_dataloader(self):
        filenames = [path.as_posix() for path in self.config["corpus"]["root"].glob("train/*.tfrecords")]
        dataset = util_tfrecords.get_dataset_from_tfrecords(
                filenames,
                feature_description='config_feature_description.pkl',
                features_to_exclude=['list_'],
                bytes_description='config_bytes_description.pkl',
                compression_type='GZIP',
                eval_mode=False,
                buffer_size_prefetch=tf.data.AUTOTUNE,
                # buffer_size_shuffle=100,
                batch_size=self.hparas_config['batch_size'],
                shuffle_seed=None,
                densify_downsample_factors=None,
                densify_jitter_indices=None,
                densify_dtype=tf.float32,
                filter_function=None,
                map_function=None,
                verbose=False)
        return dataset

    def val_dataloader(self):
        filenames = [path.as_posix() for path in self.config["corpus"]["root"].glob("valid/*.tfrecords")]
        dataset = util_tfrecords.get_dataset_from_tfrecords(
                filenames,
                feature_description='config_feature_description.pkl',
                features_to_exclude=['list_'],
                bytes_description='config_bytes_description.pkl',
                compression_type='GZIP',
                eval_mode=True,
                buffer_size_prefetch=tf.data.AUTOTUNE,
                # buffer_size_shuffle=100,
                batch_size=self.hparas_config['batch_size'],
                shuffle_seed=None,
                densify_downsample_factors=None,
                densify_jitter_indices=None,
                densify_dtype=tf.float32,
                filter_function=None,
                map_function=None,
                verbose=False)
        return dataset


trainer = Trainer(
    precision="32",
    limit_val_batches=0.0,
    num_nodes=1,
    devices=2, # was gpus=1,
    # detect_anomaly=True,
    # strategy="dp",
    accelerator="gpu",
)

config_path = "config/binaural_attn/word_task_mixed_cue_v04.yml"
config = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)

config['num_workers'] = 0
config['hparas']['batch_size'] = 80
config['hparas']['lr'] = 0.01
config['audio']['rep_kwargs']['rep_on_gpu'] = True

path_to_tf_records = Path("/om/scratch/Sat/msaddler/dataset_multitask/v04")

config['corpus']['root'] = path_to_tf_records
config['audio']['rep_kwargs']['sr'] = 44100

module = BinauralAttentionModule(config)
trainer.fit(module)


