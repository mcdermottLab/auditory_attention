import os 
from collections import namedtuple
from typing import List, Tuple, Optional, Union
import numpy as np
import torch
import torchmetrics
from torchmetrics.classification import Accuracy
from pytorch_lightning import LightningModule

import src.audio_transforms as at
import src.audio_attention_transforms as aat
import src.custom_modules as cm
from src.spatial_attn_architecture import  BinauralAuditoryAttentionCNN, BinauralControlCNN
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
        self.multi_task = self.corpora_config['task'] == 'word_and_location'

        self.corpora_name = config.get('corpora_name', False)

        # set dataset as attribute
        self.word_rec_model = False
        self.dataset = BinauralAttentionDataset 
        self.train_val_collate_fn = self._collate_fn
        v2_demean = self.audio_config.get('v2_demean', False)
        if v2_demean:
            print("Using explicit dim specification for demeaning in audio transforms")
        self.audio_transforms = at.AudioCompose([
            at.AudioToTensor(),
            at.BinauralCombineWithRandomDBSNR(low_snr=config['noise_kwargs']['low_snr'],
                                                high_snr=config['noise_kwargs']['high_snr'],
                                                v2_demean=v2_demean),
            at.BinauralRMSNormalizeForegroundAndBackground(rms_level=0.02, v2_demean=v2_demean), # 20 * np.log10(0.02/20e-6) = 60 dB SPL 
        ])
        
        if self.audio_config.get('upsample_audio', False):
            self.audio_transforms = at.AudioCompose([
                at.AudioToTensor(),
                at.BinauralCombineWithRandomDBSNR(low_snr=config['noise_kwargs']['low_snr'],
                                                  high_snr=config['noise_kwargs']['high_snr'],
                                                  v2_demean=v2_demean),
                at.BinauralRMSNormalizeForegroundAndBackground(rms_level=0.02, v2_demean=v2_demean), # 20 * np.log10(0.02/20e-6) = 60 dB SPL 
                at.Resample(**self.audio_config['upsample_kwargs'])
            ])

        self.test_step = self._test_step

        # Init Model
        # Get model architecture
        norm_first = self.model_config.get('norm_first', True)
        new_module = self.model_config.get('v08', False)
        control_arch = self.model_config.get('control_arch', False)
        self.use_backbone_arch = self.model_config.get('backbone_arch', False)
        self.backbone_with_ecdf_gains = self.model_config.get('backbone_with_ecdf_gains', False)
        self.backbone_with_learned_gains = self.model_config.get('backbone_with_learned_gains', False)
        
        self.batch_in_dataloader = (self.use_backbone_arch and not (self.backbone_with_ecdf_gains or self.backbone_with_learned_gains))
        # print(f"Batch in dataloader = {self.batch_in_dataloader}")
        self.dataset_batch_size = 1 if self.batch_in_dataloader else self.hparas_config['batch_size']
        self.dataloader_batch_size = self.hparas_config['batch_size'] if self.batch_in_dataloader else 1        

        if control_arch:
            # print("Using BinauralControlCNN")
            self.model = BinauralControlCNN(**self.model_config)
        else:
            # print("Using BinauralAuditoryAttentionCNN")
            self.model = BinauralAuditoryAttentionCNN(**self.model_config)

        # check if torch version 2 or greater - if so, compile model
        getting_acts = self.config.get('getting_acts', False)
        if not getting_acts and int(torch.__version__.split('.')[0]) >= 2 and not self.multi_task and not self.audio_config.get('upsample_audio', False):
            self.model = torch.compile(self.model, mode="default")

        ## get local rank
        # print(f"Using dataset {self.dataset.__name__}")
        # print(self.model)

        # Add input rep to model or audio transforms
        self.rep_on_gpu = self.audio_config['rep_kwargs']['rep_on_gpu']
        self.coch_gram = cm.AttnAudioInputRepresentation(**self.audio_config)

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
        self.attn_modules = [module for name, module in  self.model.model_dict.items() if 'attn' in name]
        self.bias_constraint = AttnBiasConstraint(min_val=0, max_val=1)
        self.constrain_slope = self.model_config['attn_constraints'].get('slope', False)
        if self.constrain_slope:
            self.slope_constraint = AttnSlopeConstraint(min_val=0)

    def _step(self, batch, batch_idx, step_type):
        if batch is None:
            print(f"Batch on step {batch_idx} was None")
            return None
        if self.multi_task:
            cue_features, cue_mask_ixs, loc_task_ixs, scene_features, labels = batch
        else:
            cue_features, cue_mask_ixs, scene_features, labels = batch
        
        cue_features, scene_features = self.coch_gram(cue_features, scene_features)

        # self() is self.forward()
        outputs = self(cue_features, scene_features, cue_mask_ixs)
        if self.multi_task:
            word, location = outputs
            word_label = labels[:,0]
            word_loss = self.loss_fn(word, word_label)
            # Take valid examples for location task using loc_task_mask
            location_label = labels[:,1]
            if loc_task_ixs is not None:
                location = location[loc_task_ixs, :]
                location_label = location_label[loc_task_ixs]
            loc_loss = self.loss_fn(location, location_label)
            loss = word_loss + loc_loss
            self.accuracy[step_type]['word'](word, word_label) # word accuracy
            self.accuracy[step_type]['location'](location, location_label) # location accuracy
            self.log(f"{step_type}_word_loss", word_loss.detach(), on_step=True, on_epoch=False, prog_bar=True)
            self.log(f"{step_type}_location_loss", loc_loss.detach(), on_step=True, on_epoch=False, prog_bar=True)
            self.log(f"{step_type}_loss", loss.detach(), on_step=True, on_epoch=False, prog_bar=True)
            self.log(f"{step_type}_word_acc", self.accuracy[step_type]['word'], on_step=False, on_epoch=True, prog_bar=True)
            self.log(f"{step_type}_location_acc", self.accuracy[step_type]['location'], on_step=False, on_epoch=True, prog_bar=True)
        else:
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
        model_params = [{'params': self.model.parameters()}]
        self.optimizer = opt(model_params, lr=self.hparas_config['lr'], eps=self.hparas_config['eps'])       
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

    def forward(self, cue: torch.tensor, scene: torch.tensor, cue_mask_ixs: Optional[torch.tensor] = None):
        outputs = self.model(cue, scene, cue_mask_ixs)
        # Outputs here are logits
        return outputs

    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "val")

    def _test_step(self, batch, batch_idx):
        # Not used in model simulations - test loops outside of lightning
        bg_labels = None
        if  self.audioset_bg_test or self.n_test_talkers and not self.matched_cue_level:
            signal, fg_cue, fg_labels = batch
        elif self.get_f0:
            signal, fg_cue, bg_cue, fg_labels, bg_labels, fg_f0, bg_f0 = batch
        else:
            signal, fg_cue, bg_cue, fg_labels, bg_labels = batch
        
        batch_size = len(signal)
        # self() is self.forward()  
        fg_outputs = self(fg_cue, signal) 
        fg_loss = self.loss_fn(fg_outputs, fg_labels)
        # calc foreground talker word accuracy
        if self.get_f0:
            for ix in range(batch_size):
                self.accuracy["test"](fg_outputs[ix].softmax(-1).argmax(-1).view(1,-1), fg_labels[ix].view(1,-1))
                self.log(f"ACC/test_fg_acc_eg_{ix}", self.accuracy["test"], on_step=True, on_epoch=False)

                if not self.audioset_bg_test and not self.n_test_talkers:
                    # log test confusion on tasks that have it 
                    self.accuracy['test_confusion'](fg_outputs[ix].softmax(-1).argmax(-1).view(1,-1), bg_labels[ix].view(1,-1))
                    self.log(f"test_confusion_eg_{ix}", self.accuracy['test_confusion'], on_step=True, on_epoch=False)

                self.log(f"fg_f0_eg_{ix}", fg_f0[ix], on_step=True, on_epoch=False)
                self.log(f"bg_f0_eg_{ix}", bg_f0[ix], on_step=True, on_epoch=False)
                
        else:
            # self.accuracy["test"](fg_outputs, fg_labels)
            self.log(f"ACC/test_fg_acc", self.accuracy["test"], on_step=True, on_epoch=False)

            if bg_labels != None:
                # log test confusion on tasks that have it 
                # self.accuracy['test_confusion'](fg_outputs, bg_labels)
                model_guesses = fg_outputs.log_softmax(-1).argmax(-1).view(1,-1)
                confusion = int(model_guesses in bg_labels)
                self.log(f"test_confusion", confusion, on_step=True, on_epoch=False)

        return fg_loss

    def _extract_labels(self, samples: List):
        # idx=3 is harcoded - sample in samples is list of (cue, foreground, background, label)
        return torch.tensor([sample[3] for sample in samples]).type(torch.LongTensor)

    def get_cue_mask_ixs(self, cue: torch.tensor):
        cue_flat = cue.reshape(cue.shape[0], -1)
        mask_ixs = torch.argwhere(torch.sum(torch.abs(cue_flat), dim=1) == 0).squeeze()
        if self.multi_task:
            # Only use examples in batch with cue for location task 
            loc_task_mask = torch.argwhere(torch.sum(torch.abs(cue_flat), dim=1) != 0).squeeze()
            return mask_ixs, loc_task_mask
        return mask_ixs

    def _extract_features(self, samples: List, sample_ix: Union[int, list]):
        # hardcode none for bg noise here - scenes are pre-mixed
        if self.rep_on_gpu:
            if isinstance(sample_ix, list):
                rep_features = [self.audio_transforms(sample[sample_ix[0]], sample[sample_ix[1]])[0].squeeze() for sample in samples]
            else:
                rep_features = [self.audio_transforms(sample[sample_ix], None)[0].squeeze() for sample in samples]
            features = torch.nn.utils.rnn.pad_sequence(rep_features, batch_first=True)
        else:
            # are cochleagrams & need to transpose from CxFxT -> TxFxC for pad sequence
            if isinstance(sample_ix, list):
                rep_features = [self.audio_transforms(sample[sample_ix[0]], sample[sample_ix[1]])[0].transpose(0,2) for sample in samples]
            else:
                rep_features = [self.audio_transforms(sample[sample_ix], None)[0].transpose(0,2) for sample in samples]
            features = torch.nn.utils.rnn.pad_sequence(rep_features, batch_first=True)
            features = features.transpose(1,3) # back to CxFxT for model
        return features

    def _collate_fn(self, samples: List):
        # samples is a single-element list holding a tuple batches
        samples = samples[0]
        cue_features, _ = self.audio_transforms(samples[0], None)
        if self.hparas_config.get('mask_cues', False):  # default is to not mask cues
            cue_mask_ixs, loc_task_ixs = self.get_cue_mask_ixs(cue_features)
        else:
            cue_mask_ixs, loc_task_ixs = None, None  # loc task ixs are not used if not multi_task
        scene_features, _ = self.audio_transforms(samples[1], samples[2])
        labels = torch.from_numpy(samples[3]).type(torch.LongTensor)
        if self.multi_task:
            return cue_features, cue_mask_ixs, loc_task_ixs, scene_features, labels
        return cue_features, cue_mask_ixs, scene_features, labels

    def test_collate_fn(self, samples: List):
        cue_features = self._extract_features(samples, sample_ix=0)
        cue_mask_ixs = self.get_cue_mask_ixs(cue_features)
        scene_features = self._extract_features(samples, sample_ix=[1,2])
        labels = self._extract_labels(samples)
        return cue_features, cue_mask_ixs, scene_features, labels

    def train_dataloader(self):
        self.train_dataset = self.dataset(**self.corpora_config, batch_size=self.dataset_batch_size, mode='train')
        print(f"len training set = {len( self.train_dataset )}")
        dataloader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.dataloader_batch_size ,
            num_workers=self.config['num_workers'], 
            collate_fn=self.train_val_collate_fn,
            pin_memory=True,
            # persistent_workers=True,
            shuffle=True if self.use_backbone_arch else False
        )
        return dataloader

    def val_dataloader(self):
        dataset = self.dataset(**self.corpora_config, batch_size=self.dataset_batch_size, mode='val')
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.dataloader_batch_size ,
            num_workers=self.config['num_workers'],
            collate_fn=self.train_val_collate_fn,
            shuffle=False
        )
        return dataloader

    def test_dataloader(self): # dumy placeholder no longer - fixed
        dataset = self.dataset(**self.corpora_config, mode='test')
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.hparas_config['batch_size'],
            num_workers=self.config['num_workers'],
            collate_fn=self.test_collate_fn)
        self.test_loader_len = len(dataset)
        print("Test set length = ", self.test_loader_len)
        return dataloader
