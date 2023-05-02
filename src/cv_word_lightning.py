
from collections import namedtuple
from typing import List, Tuple, Optional
import torch
import torchmetrics
from pytorch_lightning import LightningModule

import src.audio_transforms as at
import src.custom_modules as cm
from corpus.commonvoice_h5 import CommonVoiceWordTask
from src.base_aud_cnn import AuditoryCNN

# import psutil


# def get_memory_usage():
#     mem = psutil.virtual_memory()
#     return mem.used / 1024 ** 3

# Batch = namedtuple("Batch", ["features", "targets", "target_lengths"])


class CommonVoiceWordRec(LightningModule):
    def __init__(
        self,
        config: dict,
    ):
        super().__init__()
        # Make config sent to init
        self.config = config
        self.audio_config = config['audio']
        self.corpora_config = config['corpus']
        self.loader_config = config['loader']
        self.model_config = self.config['model']

        # Init dataset
        self.dataset = CommonVoiceWordTask
    
        # Init Audio Transforms
        self.audio_transforms = at.AudioCompose([
            at.AudioToTensor(),
            at.RMSNormalizeForegroundAndBackground(rms_level=0.2),
            at.UnsqueezeAudio(dim=0), 
        ])

        # Init Model        
        fc_size = self.model_config.get('fc_size', 4096)
        self.model = AuditoryCNN(self.model_config['num_words'],# vocab size
                                fc_size=fc_size) 

        # Add input rep to model or audio transforms
        self.rep_on_gpu = self.audio_config['rep_kwargs']['rep_on_gpu']
        if self.rep_on_gpu:
            self.model = cm.AttnSequentialAttacker(
                cm.AttnAudioInputRepresentation(**self.audio_config),
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
        self.test_confusion = torchmetrics.Accuracy()
        # self.test_confusion = torch.nn.ModuleDict({"fg": torchmetrics.Accuracy(),
        #                                      "bg": torchmetrics.Accuracy()})
        self.accuracy = {'train': self.train_acc,
                         'val': self.valid_acc,
                         'test': self.test_acc,
                         'test_confusion': self.test_confusion
                        }
        # Optimizer
        opt_cfg = self.config['hparas']
        opt = getattr(torch.optim, opt_cfg['optimizer'])
        model_paras = [{'params': self.model.parameters()}]
        self.optimizer = opt(model_paras,lr=opt_cfg['lr'], eps=opt_cfg['eps'])       

        
    def _step(self, batch, batch_idx, step_type):
        if batch is None:
            return None

        mixture, labels = batch
        # self() is self.forward()
        outputs = self(mixture) 
        
        loss = self.loss_fn(outputs, labels)

        self.accuracy[step_type](outputs, labels)

        self.log(f"Losses/{step_type}_loss", loss.detach(), on_step=True, on_epoch=False)        
        self.log(f"ACC/{step_type}_acc", self.accuracy[step_type], on_step=False, on_epoch=True)
        return loss
    
    def configure_optimizers(self):
        return [self.optimizer]
        
    def forward(self, mixture: torch.Tensor):
        outputs = self.model(mixture, None)
        # Outputs here are logits
        return outputs

    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "val")

    def _test_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "test")

    def _extract_labels(self, samples: List):
        return torch.tensor([sample[1] for sample in samples]).type(torch.LongTensor)

    def _extract_features(self, samples: List):
        # hardcode none for bg noise here
        if self.rep_on_gpu:
            rep_features = [self.audio_transforms(sample[0], None)[0].squeeze() for sample in samples]
            features = torch.nn.utils.rnn.pad_sequence(rep_features, batch_first=True)
        else:
            # are cochleagrams & need to transpose from CxFxT -> TxFxC for pad sequence
            rep_features = [self.audio_transforms(sample[0], None)[0].transpose(0,2) for sample in samples]
            features = torch.nn.utils.rnn.pad_sequence(rep_features, batch_first=True)
            features = features.transpose(1,3) # back to CxFxT for model
        return features

    def _collate_fn(self, samples: List):
        features = self._extract_features(samples)
        targets = self._extract_labels(samples)
        return features, targets


    def train_dataloader(self):
        dataset = self.dataset(**self.corpora_config, mode='train')
        print(f"len training set = {len(dataset)}")
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.loader_config['batch_size'],
            num_workers=self.loader_config['num_workers'], 
            collate_fn=self._collate_fn,
            pin_memory=True
        )
        return dataloader

    def val_dataloader(self):
        dataset = self.dataset(**self.corpora_config, mode='val')
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.loader_config['batch_size'],
            num_workers=self.loader_config['num_workers'],
            collate_fn=self._collate_fn,
        )
        return dataloader

    def test_dataloader(self): # dumy placeholder for now - fix 
        dataset = self.dataset(**self.corpora_config, mode='test')
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.loader_config['batch_size'],
            num_workers=self.loader_config['num_workers'],
            collate_fn=self._collate_fn)
        self.test_loader_len = len(dataset)
        print("Test set length = ", self.test_loader_len)
        return dataloader