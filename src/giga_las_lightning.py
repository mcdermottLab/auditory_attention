# Following is originally copied from PyTorch RNN-T ASR Example:
# https://github.com/pytorch/audio/tree/820b383b3b21fc06e91631a5b1e6ea1557836216/examples/asr/librispeech_emformer_rnnt

# VGG Model added instead of emformer as encoder

import json
import math
import os
from collections import namedtuple
from typing import List, Tuple

# import sentencepiece as spm
import torch
import torchaudio
import torchaudio.functional as F
from pytorch_lightning import LightningModule
from corpus.gigaspeech import GigaDataset
from src.audio import CMVN, Postprocess
from src.text import load_text_encoder
from src.asr import ASR
from src.optim import Optimizer
from src.util import cal_er
from src.decode import BeamDecoder


# from utils import GAIN, piecewise_linear_log, spectrogram_transform


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

        # Using GigaSpeech, samples already sorted on init 
        
        idx_target_lengths = [
            (idx, len(sample['transcript'].split()))
            for idx, sample in enumerate(self.base_dataset.dataset)
        ]

        assert len(idx_target_lengths) > 0

        assert max_token_limit >= idx_target_lengths[0][1]

        self.batches = _batch_by_token_count(idx_target_lengths, max_token_limit)

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
        self.feat_dim = self.config['data']['audio']['n_mels']
        self.audio_rep = torchaudio.transforms.MelSpectrogram(**self.config['data']['audio'])

        self.gigaspeech_path = self.config['data']['corpus']['path']

        self.tokenizer = load_text_encoder(self.config['data']['text']['mode'],
                                        self.config['data']['text']['vocab_file'])
        self.vocab_size = self.tokenizer.vocab_size
        self.blank_idx = 0

        self.train_data_pipeline = torch.nn.Sequential(
            CMVN(dim=1) # input is channel, time, n_mel
        )
        self.valid_data_pipeline = torch.nn.Sequential(
            CMVN(dim=1) # input is channel, time, n_mel
        )

        init_adadelta = self.config['hparas']['optimizer'] == 'Adadelta'
        self.model = ASR(self.feat_dim, self.vocab_size, init_adadelta, **
                         self.config['model'])
        model_paras = [{'params': self.model.parameters()}]
        
        # Losses
        self.seq_loss = torch.nn.CrossEntropyLoss(ignore_index=0)
        # Note: zero_infinity=False is unstable?
        self.ctc_loss = torch.nn.CTCLoss(blank=0, zero_infinity=False)
        self.optimizer = Optimizer(model_paras, **self.config['hparas'])
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=0.96, patience=0)
        self.warmup_lr_scheduler = WarmupLR(self.optimizer, 10000)
        self.step = 0 

    def _extract_labels(self, samples: List):
        targets = [self.tokenizer.encode(sample[2].upper()) for sample in samples]
        lengths = torch.tensor([len(elem) for elem in targets]).to(dtype=torch.int32)
        targets = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(elem) for elem in targets],
            batch_first=True,
            padding_value=1.0,
        ).to(dtype=torch.int32)
        return targets, lengths

    def _train_extract_features(self, samples: List):
        mel_features = [self.audio_rep(sample[1].squeeze()).transpose(1, 0) for sample in samples]
        features = torch.nn.utils.rnn.pad_sequence(mel_features, batch_first=True)
        features = self.train_data_pipeline(features)
        lengths = torch.tensor([elem.shape[0] for elem in mel_features], dtype=torch.int32)
        return features, lengths

    def _valid_extract_features(self, samples: List):
        mel_features = [self.audio_rep(sample[1].squeeze()).transpose(1, 0) for sample in samples]
        features = torch.nn.utils.rnn.pad_sequence(mel_features, batch_first=True)
        features = self.valid_data_pipeline(features)
        lengths = torch.tensor([elem.shape[0] for elem in mel_features], dtype=torch.int32)
        return features, lengths

    def _train_collate_fn(self, samples: List):
        features, feature_lengths = self._train_extract_features(samples)
        targets, target_lengths = self._extract_labels(samples)
        return Batch(features, feature_lengths, targets, target_lengths)

    def _valid_collate_fn(self, samples: List):
        features, feature_lengths = self._valid_extract_features(samples)
        targets, target_lengths = self._extract_labels(samples)
        return Batch(features, feature_lengths, targets, target_lengths)

    def _test_collate_fn(self, samples: List):
        return self._valid_collate_fn(samples), samples

    def _step(self, batch, batch_idx, step_type):
        if batch is None:
            return None

        tf_rate = self.optimizer.pre_step(self.step)
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
        if step_type == 'train':
            self.step += 1
            
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
            [self.optimizer],
            [
                {
                    "scheduler": self.lr_scheduler,
                    "monitor": "WER/val_att",
                    "interval": "epoch",
                },
                {"scheduler": self.warmup_lr_scheduler, "interval": "step"},
            ],
        )

    def forward(self, batch: Batch):
        decoder = BeamDecoder(self.model.to(self.device), None, **self.config['decode'] )
        hypotheses = decoder(batch.features.to(self.device), batch.feature_lengths.to(self.device))
        return post_process_hypos(hypotheses, self.tokenizer)[0][0]

    def training_step(self, batch: Batch, batch_idx):
        return self._step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "test")

    def train_dataloader(self):
        dataset = torch.utils.data.ConcatDataset(
                CustomDataset(
                    GigaDataset(self.gigaspeech_path, 
                                'XL_munged',
                                None, # No tokenizer - do in collate
                                1, # Batching handled by Trainer
                                True), # Sort in ascending order 
                    1000) # Max word length 
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.config['data']['corpus']['batch_size'],
            collate_fn=self._train_collate_fn,
            num_workers=10, # maybe set as config parameter?
            shuffle=True,
        )
        return dataloader

    def val_dataloader(self):
        dataset = torch.utils.data.ConcatDataset(
                CustomDataset(
                    GigaDataset(self.gigaspeech_path, 
                                'DEV_munged',
                                None, # No tokenizer - do in collate
                                1, # Batching handled by Trainer
                                True), # Sort in ascending order 
                    1000) # Max word length 
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.config['data']['corpus']['batch_size'],
            collate_fn=self._valid_collate_fn,
            num_workers=10,
        )
        return dataloader

    def test_dataloader(self):
        dataset = torchaudio.datasets.LIBRISPEECH(self.gigaspeech_path, url="test-clean")
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, collate_fn=self._test_collate_fn)
        return dataloader
