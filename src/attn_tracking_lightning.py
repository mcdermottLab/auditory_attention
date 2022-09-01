
# from collections import namedtuple
# from typing import List, Tuple, Optional
import torch
import torchmetrics
from pytorch_lightning import LightningModule
from corpus.jsinV3AttnTrackingValidation import jsinV3_attn_tracking_validation
from corpus.jsinV3_attn_tracking_multi_talker_background import jsinV3_attn_tracking_multi_talker_background

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

class AttnSlopeConstraint(object):
    def __init__(self, min_val=0):
        self.min = min_val

    def __call__(self, module):
        if hasattr(module,'slope'):
            s = module.slope.data
            module.slope.data = s.clamp(self.min) # no max -> max = inf   

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
        self.data_config = self.config['data']
        self.get_f0 = self.config.get('get_f0', False) 

        # set training dataset
        self.multi_distractor = self.data_config.get('multi_distractor', False) 
        if self.multi_distractor:
            from corpus.jsinV3_attn_multi_talker_w_audioset import jsinV3_attn_multi_talker_w_audioset
            self.train_val_dataset = jsinV3_attn_multi_talker_w_audioset
            self.bg_combine_transforms = at.AudioCompose([
                            at.AudioToTensor(),
                            at.CombineWithRandomDBSNR(low_snr=config['noise_kwargs']['low_snr'], high_snr=config['noise_kwargs']['high_snr']),
                            at.RMSNormalizeForegroundAndBackground(rms_level=0.1),
                            at.UnsqueezeAudio(dim=0),
                        ])
        else:
            from corpus.jsinV3AttnTracking import jsinV3_attn_tracking
            self.train_val_dataset = jsinV3_attn_tracking

        # set evaluation background noise params
        self.noise_only = self.data_config.get('noise_only', False) # audioset noise instead of background talker for training
        self.audioset_bg_test =  self.config.get('audioset_bg', False)
        self.n_test_distractors = self.config.get('n_distractors', False) # int or False  

        self.audio_transforms = at.AudioCompose([
            at.AudioToTensor(),
            at.CombineWithRandomDBSNR(low_snr=config['noise_kwargs']['low_snr'], high_snr=config['noise_kwargs']['high_snr']),
            at.RMSNormalizeForegroundAndBackground(rms_level=0.1),
            at.UnsqueezeAudio(dim=0),
        ])

        if self.n_test_distractors:
            self.bg_talker_transforms = at.AudioCompose([
                at.AudioToTensor(),
                at.RMSNormalizeForegroundAndBackground(rms_level=0.1)
            ])

        # Init Model
        ln_first = self.config.get('layernorm_first', False) 
        if ln_first:
            print('ln_first')
            from src.attentional_cue_model_ln_first import AuditoryCNN
        else:
            from src.attentional_cue_model import AuditoryCNN

        self.model = AuditoryCNN(self.data_config['num_words']) # vocab size

        # Add input rep to model or audio transforms
        if self.config['data']['audio']['rep_kwargs']['rep_on_gpu']:
            self.model = cm.SequentialAttacker(
                cm.AudioInputRepresentation(**self.audio_config),
                self.model
            )
            self.transforms = self.audio_transforms
        else:
            self.audio_transforms = at.AudioCompose([
                self.audio_transforms,
                at.AudioToAudioRepresentation(**self.audio_config)
            ])
            
        if self.multi_distractor:
            # list of transforms for making cochleagrams of mixtures and combining background sources 
            self.transforms = [self.audio_transforms, self.bg_combine_transforms]
        else:
            self.transforms = self.audio_transforms

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
        
        # Constraints
        self.attn_modules = [mod for name, mod in self.model._modules.items() if 'attn' in name]
        self.bias_constraint = AttnBiasConstraint(min_val=0, max_val=1)
        if 'attn_constraints' in self.config.keys():
            self.constrain_slope = self.config['attn_constraints'].get('slope', False) # False if not in config
        else:
            self.constrain_slope = False

        if self.constrain_slope:
            self.slope_constraint = AttnSlopeConstraint(min_val=0)
        
        # Learning rate scheduler 
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
            if self.constrain_slope:
                module.apply(self.slope_constraint)

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
        if  self.audioset_bg_test or self.n_test_distractors:
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
        for ix in range(batch_size):
            self.accuracy["test"](fg_outputs[ix].softmax(-1).argmax(-1).view(1,-1), fg_labels[ix].view(1,-1))
            self.log(f"ACC/test_fg_acc_eg_{ix}", self.accuracy["test"], on_step=True, on_epoch=False)

            if not self.audioset_bg_test and not self.n_test_distractors:
                # log test confusion on tasks that have it 
                self.accuracy['test_confusion'](fg_outputs[ix].softmax(-1).argmax(-1).view(1,-1), bg_labels[ix].view(1,-1))
                self.log(f"test_confusion_eg_{ix}", self.accuracy['test_confusion'], on_step=True, on_epoch=False)

            if self.get_f0:
                self.log(f"fg_f0_eg_{ix}", fg_f0[ix], on_step=True, on_epoch=False)
                self.log(f"bg_f0_eg_{ix}", bg_f0[ix], on_step=True, on_epoch=False)

        return fg_loss


    def train_dataloader(self):
        dataset = self.train_val_dataset(**self.corpora_config,
                                       mode='train',
                                       noise_only=self.noise_only,
                                       transform=self.transforms)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.loader_config['batch_size'],
            num_workers=self.config['n_jobs'],
            pin_memory=True
        )
        return dataloader

    def val_dataloader(self):
        dataset = self.train_val_dataset(**self.corpora_config,
                                       mode='val',
                                       noise_only=self.noise_only,
                                       transform=self.transforms)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.loader_config['batch_size'],
            num_workers=self.config['n_jobs']
        )
        return dataloader

    def test_dataloader(self):
        if self.n_test_distractors: 
            dataset = jsinV3_attn_tracking_multi_talker_background(**self.corpora_config, mode='test',
                                                                   transform=[self.audio_transforms, self.bg_talker_transforms],
                                                                   n_talkers=int(self.n_test_distractors))
        else:
            dataset = jsinV3_attn_tracking_validation(**self.corpora_config, mode='test', transform=self.transforms,
                                                      noise_bg=self.audioset_bg_test, get_f0=self.get_f0) 
        dataloader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=self.loader_config['batch_size'],
                                                 num_workers=self.loader_config['num_workers'])
        return dataloader
