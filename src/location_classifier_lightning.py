
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
from src.spatial_attn_architecture import BaseAuditoryNetworkForTransfer
from corpus.binaural_attention_h5 import BinauralAttentionDataset

## TO DO:  Import new dataset class


# def get_memory_usage():
#     mem = psutil.virtual_memory()
#     return mem.used / 1024 ** 3

class LocationClassifier(LightningModule):
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
        self.run_timit = self.corpora_name == 'TIMIT'

        # set dataset as attribute
        self.dataset = BinauralAttentionDataset 

        if 'v05' in self.corpora_config['root']:
            # signals are pre-combined and normalized - normalize again for certainty 
            self.audio_transforms = at.AudioCompose([
                at.AudioToTensor(),
                at.BinauralRMSNormalizeForegroundAndBackground(rms_level=0.02), # 20 * np.log10(0.02/20e-6) = 60 dB SPL 
            ])
        else:
            self.audio_transforms = at.AudioCompose([
                at.AudioToTensor(),
                at.BinauralCombineWithRandomDBSNR(low_snr=config['noise_kwargs']['low_snr'], high_snr=config['noise_kwargs']['high_snr']),
                at.BinauralRMSNormalizeForegroundAndBackground(rms_level=0.02), # 20 * np.log10(0.02/20e-6) = 60 dB SPL 
            ])

        self.test_step = self._test_step

        if self.run_timit:
            self.test_step = self.test_timit 
            self.audio_transforms = at.AudioCompose([
                at.AudioToTensor(),
                at.UnsqueezeAudio(dim=0),
            ])

        # Init Model
        # Get model architecture
        model = BaseAuditoryNetworkForTransfer(**self.model_config).load_from_checkpoint(checkpoint_path=ckpt_path, strict=False)
        self.classifier = torch.nn.Linear(self.model_config['fc_size'], config['model']['num_classes']['num_locs'])
        # check if torch version 2 or greater - if so, compile model
        self.model = torch.compile(model, mode="reduce-overhead")
        self.model.freeze()

        # Add input rep to model or audio transforms
        self.rep_on_gpu = self.audio_config['rep_kwargs']['rep_on_gpu']
        self.coch_gram = cm.AttnAudioInputRepresentation(**self.audio_config)

        # Losses
        self.loss_fn = torch.nn.CrossEntropyLoss()

        # Set up metrics
        task_key = 'num_locs'
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

    def _step(self, batch, batch_idx, step_type):
        if batch is None:
            print(f"Batch on step {batch_idx} was None")
            return None
        input_aud, labels = batch

        input_aud, _ = self.coch_gram(input_aud, None)

        # self() is self.forward()
        outputs = self(input_aud)

        loss = self.loss_fn(outputs, labels)
        self.accuracy[step_type](outputs, labels)
        self.log(f"{step_type}_loss", loss.detach(), on_step=True, on_epoch=False, prog_bar=True)
        self.log(f"{step_type}_acc", self.accuracy[step_type], on_step=False, on_epoch=True, prog_bar=True)

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


    def configure_optimizers(self):
        # Optimizer
        opt = getattr(torch.optim, self.hparas_config['optimizer'])
        model_params = [{'params': self.classifier.parameters()}] ## Use classifier params not model params 
        self.optimizer = opt(model_params, lr=self.hparas_config['lr'], eps=self.hparas_config['eps'])       
        ## New for v05 dataset - use lr Scheduler 
        # lr_schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 
        #                                                        mode='max', # monitoring val_acc, want max
        #                                                        factor=0.1,
        #                                                        patience=0.25, # wait a quarter epoch if plateaued
        #                                                        threshold=0.0001,
        #                                                        threshold_mode='rel',
        #                                                        min_lr=1e-7, 
        #                                                        verbose=True)
        # schedule = {"scheduler":lr_schedule, "monitor": self.config['val_metric']}
        return [self.optimizer]#, schedule

    def forward(self, input_aud: torch.tensor):
        with torch.nograd:
            representations = self.model(input_aud)
        outputs = self.classifier(representations)
        # Outputs here are logits
        return outputs

    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "val")

    def _test_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "test")


    def _extract_labels(self, samples: List):
        # idx=3 is harcoded - sample in samples is list of (cue, foreground, background, label)
        return torch.tensor([sample[3] for sample in samples]).type(torch.LongTensor)


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
        aud_features, _ = self.audio_transforms(samples[0], None)
        labels = torch.from_numpy(samples[3]).type(torch.LongTensor)
        return aud_features, labels

    def train_dataloader(self):
        dataset = self.dataset(**self.corpora_config, batch_size=self.hparas_config['batch_size'], mode='train')
        print(f"len training set = {len(dataset)}")
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            num_workers=self.config['num_workers'], 
            collate_fn=self._collate_fn,
            pin_memory=True,
            # persistent_workers=True,
            shuffle=False,
        )
        return dataloader

    def val_dataloader(self):
        dataset = self.dataset(**self.corpora_config, batch_size=self.hparas_config['batch_size'], mode='val')
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            num_workers=self.config['num_workers'],
            collate_fn=self._collate_fn,
            shuffle=False
        )
        return dataloader

    def test_dataloader(self): # dumy placeholder no longer - fixed
        if self.run_timit:
            from corpus.timit import TIMIT_Binaural_Compat_Prepaired
            dataset = TIMIT_Binaural_Compat_Prepaired(**self.corpora_config, mode='test')
                                        # clean_targets = self.corpora_config.get('clean_targets', False))
        else:
            dataset = self.dataset(**self.corpora_config, mode='test')

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.hparas_config['batch_size'],
            num_workers=self.config['num_workers'],
            collate_fn=self.test_collate_fn)
        self.test_loader_len = len(dataset)
        print("Test set length = ", self.test_loader_len)
        return dataloader