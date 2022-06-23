
from collections import namedtuple
from typing import List, Tuple, Optional
import torch
import torchaudio
import torchmetrics
import numpy as np 
import torchaudio.functional as F
from pytorch_lightning import LightningModule
from src.attentional_cue_model import AuditoryCNN
from corpus.jsinV3AttnTracking import jsinV3_attn_tracking
from src.util import cal_er
import src.audio_transforms as at
import src.custom_modules as cm 
# import psutil


# def get_memory_usage():
#     mem = psutil.virtual_memory()
#     return mem.used / 1024 ** 3


class AttnBiasConstraint(object):
    def __init__(self, min_val=0, max_val=1):
        self.min = min_val
        self.max = max_val
        
    def __call__(self, module):
        if hasattr(module,'bias'):
            b = module.bias.data
            module.bias.data = b.clamp(self.min, self.max)
            

class AttentionalTrackingModule(LightningModule):
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

        self.model = AuditoryCNN(self.data_config['num_words']) # vocab size
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
        
        self.attn_modules = [mod for name, mod in self.model._modules.items() if 'attn' in name]
        
        self.bias_constraint = AttnBiasConstraint(min_val=0, max_val=1)
        
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
        

    def _step(self, batch, batch_idx, step_type):
        if batch is None:
            return None
        mixture, cue, labels = batch
        # self() is self.forward()
        outputs = self(cue, mixture) 
        
        loss = self.loss_fn(outputs, labels)
        # calc accuracy
        self.accuracy[step_type](outputs, labels)

        self.log(f"Losses/{step_type}_loss", loss.detach(), on_step=True, on_epoch=False)        
        self.log(f"ACC/{step_type}_acc", self.accuracy[step_type], on_step=False, on_epoch=True)
        return loss
    
    def on_before_zero_grad(self, *args, **kwargs):
        for module in self.attn_modules:
            module.apply(self.bias_constraint)

    def configure_optimizers(self):
        return [self.optimizer]
        
    def forward(self, cue: torch.Tensor, mixture: torch.Tensor):
        outputs = self.model(cue, mixture)
        # Outputs here are logits
        return outputs

    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "test")

    def train_dataloader(self):
        dataset = jsinV3_attn_tracking(**self.corpora_config, train=True, transform=self.audio_transforms)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.loader_config['batch_size'],
            num_workers=self.config['n_jobs'],
            pin_memory=True
        )
        return dataloader

    def val_dataloader(self):
        dataset = jsinV3_attn_tracking(**self.corpora_config, train=False, transform=self.audio_transforms)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.loader_config['batch_size'],
            num_workers=self.config['n_jobs']
        )
        return dataloader

    def test_dataloader(self):
        dataset = jsinV3_attn_tracking(**self.corpora_config, train=False, transform=self.audio_transforms) 
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)
        return dataloader
