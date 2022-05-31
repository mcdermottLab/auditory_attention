
from collections import namedtuple
from typing import List, Tuple, Optional
import torch
import torchaudio
import torchmetrics
import numpy as np 
import torchaudio.functional as F
from pytorch_lightning import LightningModule
from src.module import CNN2DClassifier
from corpus.jsinV3DataLoader_precombined import jsinV3_precombined
from src.util import cal_er
import src.audio_transforms as at
import src.custom_modules as cm 


Batch = namedtuple("Batch", ["features", "targets"])


class CochWordRecModule(LightningModule):
    def __init__(
        self,
        config: dict,
    ):
        super().__init__()
        # Make config sent to init
        self.config = config
        self.audio_config = config['data']['audio']
        self.corpora_config = config['data']['corpus']
        self.loader_config = config['data']['loader']

        self.audio_transforms = at.AudioCompose([
            at.AudioToTensor(),
            at.CombineWithRandomDBSNR(low_snr=-10, high_snr=10),
            at.RMSNormalizeForegroundAndBackground(rms_level=0.1),
            at.UnsqueezeAudio(dim=0),
        ])

        self.data_config = self.config['data']

        self.model = CNN2DClassifier(self.data_config['num_words'], # vocab size
                                     self.audio_config['rep_kwargs']['n_channels'],
                                     1, # Num input channels is 1 for our models 
                                     **self.config['model']) # populate from config

        if self.config['data']['audio']['rep_kwargs']['rep_on_gpu']:
            self.model = cm.SequentialAttacker(
                cm.AudioInputRepresentation(**self.audio_config),
                self.model
            )
        else:
            self.audio_transforms = at.AudioCompose([
                self.audio_transforms,
                at.AudioToAudioRepresentation(**self.audio_config)
            ])

        # Losses
        self.loss_fn = torch.nn.CrossEntropyLoss()

                # Set up metrics
        self.train_acc = torchmetrics.Accuracy()
        self.valid_acc = torchmetrics.Accuracy()
        self.test_acc = torchmetrics.Accuracy()
        self.accuracy = {'train': self.train_acc,
                         'val': self.valid_acc,
                         'test': self.test_acc}
        # Optimizer
        opt_cfg = self.config['hparas']
        opt = getattr(torch.optim, opt_cfg['optimizer'])
        model_paras = [{'params': self.model.parameters()}]
        
        if opt_cfg['lr_scheduler'] == 'warmup':
            warmup_step = 4000.0
            init_lr = opt_cfg['lr']
            update_rule= lambda step: init_lr * warmup_step ** 0.5 * \
                min((step+1)*warmup_step**-1.5, (step+1)**-0.5)
            self.optimizer = opt(model_paras, lr=1.0)
            self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, update_rule) 
        else:
            self.lr_scheduler = None
            self.optimizer = opt(model_paras,lr=opt_cfg['lr'], eps=opt_cfg['eps'])       
            

    def _extract_labels(self, samples: List):
        targets = torch.tensor([elem[-1] for elem in samples]).type(torch.LongTensor)
        return targets

    def _train_extract_features(self, samples: List):
        features = [self.audio_transforms(*sample[:-1])[0] for sample in samples]
        features = torch.nn.utils.rnn.pad_sequence(features, batch_first=True)
        return features

    def _valid_extract_features(self, samples: List):
        features = [self.audio_transforms(*sample[:-1])[0] for sample in samples]
        features = torch.nn.utils.rnn.pad_sequence(features, batch_first=True)
        return features

    def _train_collate_fn(self, samples: List):
        features = self._train_extract_features(samples)
        targets = self._extract_labels(samples)
        return Batch(features, targets)

    def _valid_collate_fn(self, samples: List):
        features = self._valid_extract_features(samples)
        targets = self._extract_labels(samples)
        return Batch(features, targets)

    def _test_collate_fn(self, samples: List):
        return self._valid_collate_fn(samples)


    def _step(self, batch, batch_idx, step_type):
        if batch is None:
            return None
        # self() is self.forward()
        outputs = self(batch) 
        
        loss = self.loss_fn(outputs, batch.targets)
        # calc accuracy
        self.accuracy[step_type](outputs, batch.targets)
        # log loss and acc
        self.log(f"Losses/{step_type}_loss", loss, on_step=True, on_epoch=True)        
        self.log(f"ACC/{step_type}_acc", self.accuracy[step_type], on_step=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return [self.optimizer]
        
    def forward(self, batch: Batch):
        outputs = self.model(batch.features)
        # Outputs here are top 1 predictions
        return outputs

    def training_step(self, batch: Batch, batch_idx):
        return self._step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "test")

    def train_dataloader(self):
        dataset = jsinV3_precombined(**self.corpora_config, train=True)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.loader_config['batch_size'],
            num_workers=self.config['n_jobs'], 
            collate_fn=self._train_collate_fn,
            pin_memory=True
        )
        return dataloader

    def val_dataloader(self):
        dataset = jsinV3_precombined(**self.corpora_config, train=False)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.loader_config['batch_size'],
            num_workers=self.config['n_jobs'],
            collate_fn=self._valid_collate_fn,
        )
        return dataloader

    def test_dataloader(self):
        dataset = jsinV3_precombined(**self.corpora_config, train=False)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, collate_fn=self._test_collate_fn)
        return dataloader
