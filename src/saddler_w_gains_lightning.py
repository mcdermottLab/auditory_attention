import os 
from collections import namedtuple
from typing import List, Tuple, Optional, Union
import numpy as np
import torch
from torchmetrics.classification import Accuracy
from pytorch_lightning import LightningModule

import src.audio_transforms as at
import src.audio_attention_transforms as aat
import src.custom_modules as cm
from src.spatial_attn_architecture import SaddlerBackBoneLearnedGains
from corpus.binaural_attention_h5 import BinauralAttentionDataset


class AttnBiasConstraint(object):
    def __init__(self, min_val=0, max_val=1):
        self.min = min_val
        self.max = max_val
        
    def __call__(self, module):
        if hasattr(module,'bias'):
            b = module.bias.data
            module.bias.data = b.clamp(self.min, self.max)

class AttnSlopeConstraint(object):
    def __init__(self, min_val=0):
        self.min = min_val

    def __call__(self, module):
        if hasattr(module,'slope'):
            s = module.slope.data
            module.slope.data = s.clamp(self.min) # no max -> max = inf   

class SaddlerBackBoneModule(LightningModule):
    def __init__(
        self,
        config: dict,
    ):
        super().__init__()
        # Make config sent to init
        self.config = config
        self.corpora_config = config['corpus']
        self.hparas_config = config['hparas']
        self.multi_task = self.corpora_config['task'] == 'word_and_location'
        self.corpora_name = config.get('corpora_name', False)

        # set dataset as attribute
        self.word_rec_model = False
        self.dataset = BinauralAttentionDataset 
        self.train_val_collate_fn = self._collate_fn

        self.audio_transforms = at.AudioCompose([
            at.AudioToTensor(),
            at.Resample(orig_freq=44_100, new_freq=50_000),
            at.BinauralCombineWithRandomDBSNR(low_snr=config['noise_kwargs']['low_snr'],
                                                high_snr=config['noise_kwargs']['high_snr'],
                                                v2_demean=True),
            at.BinauralRMSNormalizeForegroundAndBackground(rms_level=0.02, v2_demean=True), # 20 * np.log10(0.02/20e-6) = 60 dB SPL 
        ])

        # Init Model
        # Get model architecture

        # self.batch_in_dataloader = (self.use_backbone_arch and not (self.backbone_with_ecdf_gains or self.backbone_with_learned_gains))
        # print(f"Batch in dataloader = {self.batch_in_dataloader}")
        # self.dataset_batch_size = 1 if self.batch_in_dataloader else self.hparas_config['batch_size']
        # self.dataloader_batch_size = self.hparas_config['batch_size'] if self.batch_in_dataloader else 1        
        self.dataset_batch_size = self.hparas_config['batch_size']
        self.dataloader_batch_size =  1        
        
        self.model = SaddlerBackBoneLearnedGains(dir_model=config['model']['dir_model'])

        print(f"Using dataset {self.dataset.__name__}")
        # print(self.model)

        # Losses
        self.loss_fn = torch.nn.CrossEntropyLoss()

        # Set up metrics
        if self.multi_task:
            self.train_acc = torch.nn.ModuleDict({'word':Accuracy(task="multiclass", num_classes=config['model']['num_classes']['num_words']), 
                                                  'location':Accuracy(task="multiclass", num_classes=config['model']['num_classes']['num_locs'])})
            self.valid_acc = torch.nn.ModuleDict({'word':Accuracy(task="multiclass", num_classes=config['model']['num_classes']['num_words']),
                                                 'location':Accuracy(task="multiclass", num_classes=config['model']['num_classes']['num_locs'])})
            self.test_acc = torch.nn.ModuleDict({'word':Accuracy(task="multiclass", num_classes=config['model']['num_classes']['num_words']),
                                                 'location':Accuracy(task="multiclass", num_classes=config['model']['num_classes']['num_locs'])})
            self.test_confusion = torch.nn.ModuleDict({'word': Accuracy(task="multiclass", num_classes=config['model']['num_classes']['num_words']),
                                                 'location': Accuracy(task="multiclass", num_classes=config['model']['num_classes']['num_locs'])})
        else:
            task_key = 'num_words' if self.corpora_config['task'] == 'word' else "num_locs"
            self.train_acc = Accuracy(task="multiclass", num_classes=config['model']['num_classes'][task_key]).to(self.device)
            self.valid_acc = Accuracy(task="multiclass", num_classes=config['model']['num_classes'][task_key]).to(self.device)
            self.test_acc = Accuracy(task="multiclass", num_classes=config['model']['num_classes'][task_key]).to(self.device)
            self.test_confusion = Accuracy(task="multiclass", num_classes=config['model']['num_classes'][task_key]).to(self.device)
        self.accuracy = {'train': self.train_acc,
                         'val': self.valid_acc,
                         'test': self.test_acc,
                         'test_confusion': self.test_confusion
                        }

        # Constraints
        self.attn_modules = [module for name, module in  self.model.gains.items() if 'attn' in name]
        self.bias_constraint = AttnBiasConstraint(min_val=0, max_val=1)
        self.constrain_slope = True
        self.slope_constraint = AttnSlopeConstraint(min_val=0)

    def _step(self, batch, batch_idx, step_type):
        if batch is None:
            print(f"Batch on step {batch_idx} was None")
            return None
        if self.multi_task:
            cue_features, cue_mask_ixs, loc_task_ixs, scene_features, labels = batch
        else:
            cue_features, cue_mask_ixs, scene_features, labels = batch
        
        labels = labels + 1 # push into range of pre-trained model 
        # self() is self.forward()
        outputs = self(cue_features, scene_features, cue_mask_ixs)

        loss = self.loss_fn(outputs, labels)
        self.accuracy[step_type](outputs, labels)
        self.log(f"{step_type}_loss", loss.detach(), on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(f"{step_type}_acc", self.accuracy[step_type], on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        return loss

    def on_before_optimizer_step(self, _):
        def _get_grad_norm(params, scale=1):
            """Compute grad norm given a gradient scale."""
            total_norm = 0.0
            for p in params:
                if p.grad is not None:
                    param_norm = (p.grad.detach().data / scale).norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm**0.5
            return total_norm
        grad_norm = _get_grad_norm(self.model.parameters())
        self.log("grad_norm", torch.tensor(grad_norm), prog_bar=True, on_step=True, on_epoch=False)

    def on_before_zero_grad(self, *args, **kwargs):
        for module in self.attn_modules:
            module.apply(self.bias_constraint)
            if self.constrain_slope:
                module.apply(self.slope_constraint)

    def configure_optimizers(self):
        # Optimizer
        opt = getattr(torch.optim, self.hparas_config['optimizer'])
        model_params = [{'params': self.model.gains.parameters()}]
        self.optimizer = opt(model_params, lr=self.hparas_config['lr'], eps=self.hparas_config['eps'])       
        ## New for v05 dataset - use lr Scheduler 
        if self.hparas_config.get('use_scheduler', False):
            print(f"Using learning rate schedule: {self.hparas_config['scheduler']['type']}")
            if self.hparas_config['scheduler']['type'] == "OneCycleLR":
                # quick hack to get len of training set 
                loader = self.train_dataloader()
                dataset_len = len(self.train_dataset)
                del loader 
                scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer,
                                                                max_lr=self.hparas_config['scheduler']['max_lr'],
                                                                epochs=self.hparas_config['epochs'],
                                                                three_phase=False,
                                                                # verbose=True,
                                                                steps_per_epoch=dataset_len//self.config['ngpus'])

                lr_scheduler = {'scheduler': scheduler, 'interval': 'step'}
            elif self.hparas_config['scheduler']['type'] == "ReduceLROnPlateau":
                scheduler =  torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                                        mode=self.hparas_config['scheduler']['mode'],
                                                                        patience=self.hparas_config['scheduler']['patience'],
                                                                        factor=self.hparas_config['scheduler']['factor'],
                                                                        )
                lr_scheduler = {'scheduler': scheduler,
                            'monitor': self.config['val_metric'], 
                            'interval': 'epoch',
                            'frequency': 1 
                            }

            return {'optimizer': self.optimizer, 'lr_scheduler': lr_scheduler}

        return [self.optimizer]

    def forward(self, cue: torch.tensor, scene: torch.tensor, cue_mask_ixs: torch.tensor):
        outputs = self.model(cue, scene, cue_mask_ixs)
        # Outputs here are logits
        return outputs

    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "val")


    def get_cue_mask_ixs(self, cue: torch.tensor):
        cue_flat = cue.reshape(cue.shape[0], -1)
        mask_ixs = torch.argwhere(torch.sum(torch.abs(cue_flat), dim=1) == 0).squeeze()
        if self.multi_task:
            # Only use examples in batch with cue for location task 
            loc_task_mask = torch.argwhere(torch.sum(torch.abs(cue_flat), dim=1) != 0).squeeze()
            return mask_ixs, loc_task_mask
        return mask_ixs


    def _collate_fn(self, samples: List):
        # samples is a single-element list holding a tuple batches
        samples = samples[0]
        cue_features, _ = self.audio_transforms(samples[0], None)
        if self.hparas_config.get('mask_cues', False):  # default is to not mask cues
            cue_mask_ixs, loc_task_ixs = self.get_cue_mask_ixs(cue_features)
        else:
            cue_mask_ixs, loc_task_ixs = None, None  # loc task ixs are not used if not multi_task
        scene_features, _ = self.audio_transforms(samples[1], samples[2])
        # permute to match saddler input dims 
        # -> batch x channels x time -> batch x time x channels
        cue_features = cue_features.permute(0, 2, 1)
        scene_features = scene_features.permute(0, 2, 1)
        labels = torch.from_numpy(samples[3]).type(torch.LongTensor)
        return cue_features, None, scene_features, labels


    def train_dataloader(self):
        self.train_dataset = self.dataset(**self.corpora_config, batch_size=self.dataset_batch_size, mode='train')
        print(f"len training set = {len( self.train_dataset )}")
        dataloader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.dataloader_batch_size ,
            num_workers=self.config['num_workers'], 
            collate_fn=self._collate_fn,
            pin_memory=True,
            # persistent_workers=True,
            shuffle=False # True if self.use_backbone_arch else False
        )
        return dataloader

    def val_dataloader(self):
        dataset = self.dataset(**self.corpora_config, batch_size=self.dataset_batch_size, mode='val')
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.dataloader_batch_size ,
            num_workers=self.config['num_workers'],
            collate_fn=self._collate_fn,
            shuffle=False
        )
        return dataloader
