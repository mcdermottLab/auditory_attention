# Following is originally copied from PyTorch RNN-T ASR Example:
# https://github.com/pytorch/audio/tree/820b383b3b21fc06e91631a5b1e6ea1557836216/examples/asr/librispeech_emformer_rnnt

# VGG Model added instead of emformer as encoder

import json
import math
import os
from collections import namedtuple
from typing import List, Tuple
import numpy as np 
# import sentencepiece as spm
import torch
import torchaudio
import torchaudio.functional as F
from pytorch_lightning import LightningModule
from src.text import load_text_encoder
from src.asr import ASR
from src.audio import create_transform
from src.optim import Optimizer
from src.util import cal_er
from src.decode import BeamDecoder


# from utils import GAIN, piecewise_linear_log, spectrogram_transform

# mel_spec_transform = torchaudio.compliance.kaldi.fbank


# def spectrogram_transform(wav):
#     return mel_spec_transform(wav, num_mel_bins=40, sample_frequency=16000,channel=-1,
#                        frame_length=25, frame_shift=10)


Batch = namedtuple("Batch", ["features", "feature_lengths", "targets", "target_lengths"])


def _batch_by_token_count(idx_target_lengths, token_limit):
    batches = []
    current_batch = []
    current_token_count = 0
    for idx, target_length in idx_target_lengths:
        if current_token_count + target_length > token_limit:
            batches.append(current_batch)
            current_batch = [idx]
            current_token_count = target_length
        else:
            current_batch.append(idx)
            current_token_count += target_length

    if current_batch:
        batches.append(current_batch)

    return batches


class CustomDataset(torch.utils.data.Dataset):
    r"""Sort samples by target length and batch to max token count."""

    def __init__(self, base_dataset, max_token_limit):
        super().__init__()
        self.base_dataset = base_dataset

        fileid_to_target_length = {}
        idx_target_lengths = [
            (idx, self._target_length(fileid, fileid_to_target_length))
            for idx, fileid in enumerate(self.base_dataset._walker)
        ]

        assert len(idx_target_lengths) > 0

        idx_target_lengths = sorted(idx_target_lengths, key=lambda x: x[1], reverse=True)

        assert max_token_limit >= idx_target_lengths[0][1]

        self.batches = _batch_by_token_count(idx_target_lengths, max_token_limit)

    def _target_length(self, fileid, fileid_to_target_length):
        if fileid not in fileid_to_target_length:
            speaker_id, chapter_id, _ = fileid.split("-")

            file_text = speaker_id + "-" + chapter_id + self.base_dataset._ext_txt
            file_text = os.path.join(self.base_dataset._path, speaker_id, chapter_id, file_text)

            with open(file_text) as ft:
                for line in ft:
                    fileid_text, transcript = line.strip().split(" ", 1)
                    fileid_to_target_length[fileid_text] = len(transcript)

        return fileid_to_target_length[fileid]

    def __getitem__(self, idx):
        return [self.base_dataset[subidx] for subidx in self.batches[idx]]

    def __len__(self):
        return len(self.batches)

    

class TimeMasking(torchaudio.transforms._AxisMasking):
    def __init__(self, time_mask_param: int, min_mask_p: float, iid_masks: bool = False) -> None:
        super(TimeMasking, self).__init__(time_mask_param, 2, iid_masks)
        self.min_mask_p = min_mask_p

    def forward(self, specgram: torch.Tensor, mask_value: float = 0.0) -> torch.Tensor:
        if self.iid_masks and specgram.dim() == 4:
            mask_param = min(self.mask_param, self.min_mask_p * specgram.shape[self.axis + 1])
            return F.mask_along_axis_iid(specgram, mask_param, mask_value, self.axis + 1)
        else:
            mask_param = min(self.mask_param, self.min_mask_p * specgram.shape[self.axis])
            return F.mask_along_axis(specgram, mask_param, mask_value, self.axis)


class FunctionalModule(torch.nn.Module):
    def __init__(self, functional):
        super().__init__()
        self.functional = functional

    def forward(self, input):
        return self.functional(input)


class GlobalStatsNormalization(torch.nn.Module):
    def __init__(self, global_stats_path):
        super().__init__()

        with open(global_stats_path) as f:
            blob = json.loads(f.read())

        self.mean = torch.tensor(blob["mean"])
        self.invstddev = torch.tensor(blob["invstddev"])

    def forward(self, input):
        return (input - self.mean) * self.invstddev
    
    
class CMVN(torch.nn.Module):
    def __init__(self, mode="global", dim=2, eps=1e-10):
        # `torchaudio.load()` loads audio with shape [channel, feature_dim, time]
        # so perform normalization on dim=2 by default
        super(CMVN, self).__init__()

        if mode != "global":
            raise NotImplementedError(
                "Only support global mean variance normalization.")

        self.mode = mode
        self.dim = dim
        self.eps = eps

    def forward(self, x):
        if self.mode == "global":
            return (x - x.mean(self.dim, keepdim=True)) / (self.eps + x.std(self.dim, keepdim=True))

    def extra_repr(self):
        return "mode={}, dim={}, eps={}".format(self.mode, self.dim, self.eps)

    
class WarmupLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_updates, last_epoch=-1, verbose=False):
        self.warmup_updates = warmup_updates
        super().__init__(optimizer, last_epoch=last_epoch, verbose=verbose)

    def get_lr(self):
        return [(min(1.0, self._step_count / self.warmup_updates)) * base_lr for base_lr in self.base_lrs]


def post_process_hypos(
    hypos: List[int], tokenizer: object
) -> List[Tuple[str, float, List[int], List[int]]]:
    post_process_remove_list = [
        tokenizer.unk_idx(),
        tokenizer.eos_idx(),
        tokenizer.pad_idx(),
    ]
    filtered_hypo_tokens = [
        [token_index for token_index in h[1:] if token_index not in post_process_remove_list] for h in hypos
    ]
    hypos_str = [tokenizer.decode(s) for s in filtered_hypo_tokens]
    hypos_ali = [h.alignment[1:] for h in hypos]
    hypos_ids = [h.tokens[1:] for h in hypos]
    hypos_score = [[math.exp(h.score)] for h in hypos]

    nbest_batch = list(zip(hypos_str, hypos_score, hypos_ali, hypos_ids))

    return nbest_batch


class LASModule(LightningModule):
    def __init__(
        self,
        config: dict,
    ):
        super().__init__()
        # Make config sent to init
        self.config = config
        self.librispeech_path = self.config['data']['corpus']['path']

        # set training batch size params
        self.ascending = self.config['hparas']['curriculum'] > 0
        self.bucketing = self.config['data']['corpus']['bucketing']
        # mel spec and model param
        self.feat_dim = self.config['data']['audio']['feat_dim']

        self.tokenizer = load_text_encoder(self.config['data']['text']['mode'],
                                        self.config['data']['text']['vocab_file'])
        self.vocab_size = self.tokenizer.vocab_size
        self.blank_idx = 0

        self.data_pipeline, _ = create_transform(config['data']['audio'])

        init_adadelta = self.config['hparas']['optimizer'] == 'Adadelta'

        self.model = ASR(self.feat_dim, self.vocab_size, init_adadelta, **
                         self.config['model'])
        
        # Losses
        self.seq_loss = torch.nn.CrossEntropyLoss(ignore_index=0)
        # Note: zero_infinity=False is unstable?
        self.ctc_loss = torch.nn.CTCLoss(blank=0, zero_infinity=False)

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
        elif opt_cfg['lr_scheduler'] == 'plateau':
            self.optimizer = opt(model_paras,lr=opt_cfg['lr'], eps=opt_cfg['eps'])
            self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,'min', patience=1)
        else:
            self.lr_scheduler = None
            self.optimizer = opt(model_paras,lr=opt_cfg['lr'], eps=opt_cfg['eps'])       
            
        self.step = 0 

        # Teacher Forcing 
        self.tf_start = opt_cfg['tf_start']
        self.tf_end = opt_cfg['tf_end']
        self.tf_step = opt_cfg['tf_step']
        
    def _tf_rate(self,step):
        return max(self.tf_end,
                   self.tf_start-(self.tf_start-self.tf_end)*step/self.tf_step)
    
    def _pre_step(self, step):
        if self.lr_scheduler is not None:
            cur_lr = self.lr_scheduler.get_last_lr()[0]
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = cur_lr
#         self.opt.zero_grad()
        return self._tf_rate(step)

    def _extract_labels(self, samples: List):
        if self.bucketing and len(samples) == 1:
            samples = samples[0]
        targets = [self.tokenizer.encode(sample[2].upper()) for sample in samples]
        lengths = torch.tensor([len(elem) for elem in targets]).to(dtype=torch.int32)
        targets = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(elem) for elem in targets],
            batch_first=True).type(torch.LongTensor)
        return targets, lengths

    def _extract_features(self, samples: List):
        if self.bucketing and len(samples) == 1:
            samples = samples[0]
        mel_features = [self.data_pipeline(sample[0]) for sample in samples]
        features = torch.nn.utils.rnn.pad_sequence(mel_features, batch_first=True)
        lengths = torch.tensor([elem.shape[0] for elem in mel_features], dtype=torch.int32)
        return features, lengths

    def _train_collate_fn(self, samples: List):
        features, feature_lengths = self._extract_features(samples)
        targets, target_lengths = self._extract_labels(samples)
        return Batch(features, feature_lengths, targets, target_lengths)

    def _valid_collate_fn(self, samples: List):
        return self._train_collate_fn(samples)

    def _test_collate_fn(self, samples: List):
        return self._valid_collate_fn(samples), samples

    def _step(self, batch, batch_idx, step_type):
        if batch is None:
            return None
        total_loss = 0
        tf_rate = self._pre_step(self.step)
        ctc_output, encode_len, att_output, att_align, dec_state = \
            self.model(batch.features,
                       batch.feature_lengths,
                       max(batch.target_lengths),
                       tf_rate=tf_rate,
                       teacher=batch.targets, 
                       get_dec_state=False) # hard code not using embedding
        if ctc_output is not None:
            ctc_loss = self.ctc_loss(ctc_output.transpose(
                            0, 1), batch.targets,
                            encode_len, batch.target_lengths)
            total_loss += ctc_loss * self.model.ctc_weight
        if att_output is not None:
            b, t, _ = att_output.shape
            att_loss = self.seq_loss(
                att_output.view(b*t, -1), batch.targets.view(-1))
            total_loss += att_loss*(1-self.model.ctc_weight)

        self.log(f"Losses/{step_type}_loss", total_loss, on_step=True, on_epoch=True)        
            
        if step_type != 'train':
            wer_att = cal_er(self.tokenizer, att_output, batch.targets)
            wer_ctc = cal_er(self.tokenizer, ctc_output, batch.targets, ctc=True)
            if att_output is not None:
                self.log(f"WER/{step_type}_att", wer_att, on_step=True, on_epoch=True)
            if ctc_output is not None:
                self.log(f"WER/{step_type}_ctc", wer_ctc, on_step=True, on_epoch=True)

        return total_loss

    def configure_optimizers(self):
        return (
            [self.optimizer]# ,
#             [
#                 {
#                     "scheduler": self.lr_scheduler,
#                     "monitor": "WER/val_att",
#                     "interval": "epoch",
#                 }
               
#             ],
        )

    def forward(self, batch: Batch):
        decoder = BeamDecoder(self.model.to(self.device), None, **self.config['decode'] )
        hypotheses = decoder(batch.features.to(self.device), batch.feature_lengths.to(self.device))
        return post_process_hypos(hypotheses, self.tokenizer)[0][0]

    def training_step(self, batch: Batch, batch_idx):
        self.step += 1
        return self._step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "test")

    def train_dataloader(self):
        dataset = torch.utils.data.ConcatDataset(
            [
                CustomDataset(
                    torchaudio.datasets.LIBRISPEECH(os.path.join(self.librispeech_path, 'train-clean-360/'),
                                                    url="train-clean-360"),
                    1000,
                ),
                CustomDataset(
                    torchaudio.datasets.LIBRISPEECH(os.path.join(self.librispeech_path, 'train-clean-100/'),
                                                    url="train-clean-100"),
                    1000,
                ),
                CustomDataset(
                    torchaudio.datasets.LIBRISPEECH(os.path.join(self.librispeech_path, 'train-other-500/'),
                                                    url="train-other-500"),
                    1000,
                ),
            ]
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=None,#self.config['data']['corpus']['batch_size'],
            collate_fn=self._valid_collate_fn,
            num_workers=self.config['n_jobs'],
            shuffle=True,
        )
        return dataloader

    def val_dataloader(self):
        dataset = torch.utils.data.ConcatDataset(
            [
                CustomDataset(
                    torchaudio.datasets.LIBRISPEECH(os.path.join(self.librispeech_path, 'dev-clean'), url="dev-clean"),
                    1000,
                ),
                CustomDataset(
                    torchaudio.datasets.LIBRISPEECH(os.path.join(self.librispeech_path, 'dev-other/'), url="dev-other"),
                    1000,
                ),
            ]
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=None,#self.config['data']['corpus']['batch_size'],
            collate_fn=self._valid_collate_fn,
            num_workers=self.config['n_jobs']
        )
        return dataloader

    def test_dataloader(self):
        dataset = torchaudio.datasets.LIBRISPEECH(os.path.join(self.librispeech_path, 'test-clean/'), url="test-clean")
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, collate_fn=self._test_collate_fn)
        return dataloader