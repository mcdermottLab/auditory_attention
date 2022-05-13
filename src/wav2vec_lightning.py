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
import fairseq
import torchaudio
import torchaudio.functional as F
from pytorch_lightning import LightningModule
from corpus.gigaspeech import GigaDataset
from src.text import load_text_encoder
from src.audio import create_transform
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
        # Using GigaSpeech, samples already sorted in munged csv 
        
        # use pre-saved info in csv loaded by huggingface 
        # don't unpack to list - we only need this info once to make batches 
        idx_target_lengths = zip(self.base_dataset.dataset['index'],
                                 self.base_dataset.dataset['n_words'])
        
        assert len(self.base_dataset.dataset) > 0

        assert max_token_limit >= self.base_dataset.dataset['n_words'][0]

        self.batches = _batch_by_token_count(idx_target_lengths, max_token_limit)
        print("Done Initializing Data")

    def __getitem__(self, idx):
        return [self.base_dataset[subidx] for subidx in self.batches[idx]]

    def __len__(self):
        return len(self.batches)

    
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


class wav2vecModel(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        wav2vec, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task([config['wav2vec_path']])
        self.wav2vec = wav2vec[0]  
        self.decoder = getattr(torch.nn, config['model']['layer_type'])(**config['model']['layer_params'])

    def forward(self, input, input_lens):
        input = self.wav2vec.feature_extractor(input)
        input = self.wav2vec.feature_aggregator(input)
       
        input = input.transpose(1,2) # BxDxT -> BxTxD   
        input = self.decoder(input)
        encoded_len = input_lens // 160 - 2 # 160 is wav2vec downsample rate - 10ms output stride 
#        encoded_len = torch.div(input_lens, 160.26, rounding_mode='floor').to(torch.long)
        
        return input, encoded_len


class wav2vecModule(LightningModule):
    def __init__(
        self,
        config: dict,
    ):
        super().__init__()
        # Make config sent to init
        self.config = config
        self.gigaspeech_path = self.config['data']['corpus']['path']

        # set training batch size params
        self.ascending = self.config['hparas']['curriculum'] > 0
        self.bucketing = self.config['data']['corpus']['bucketing']
        # mel spec and model param

        self.tokenizer = load_text_encoder(self.config['data']['text']['mode'],
                                        self.config['data']['text']['vocab_file'])
        self.vocab_size = self.tokenizer.vocab_size
        self.blank_idx = 0
        self.config['model']['layer_params']['out_features'] = self.vocab_size 
        # self.data_pipeline, _ = create_transform(config['data']['audio'])
        

        # init wav2vec 
        self.model = wav2vecModel(config)
        
        # freeze wav2vec params
        for para in self.model.wav2vec.parameters():
            para.requires_grad = False
        
        # get trainable params
        
        trainable_params = [{'params': self.model.decoder.parameters()}] 
        
        # Losses
        # Note: zero_infinity=False is unstable?
        self.ctc_loss = torch.nn.CTCLoss(blank=0, zero_infinity=False)

        # Optimizer
        opt_cfg = self.config['hparas']
        opt = getattr(torch.optim, opt_cfg['optimizer'])
        
        if opt_cfg['lr_scheduler'] == 'warmup':
            warmup_step = 4000.0
            init_lr = opt_cfg['lr']
            update_rule= lambda step: init_lr * warmup_step ** 0.5 * \
                min((step+1)*warmup_step**-1.5, (step+1)**-0.5)
            self.optimizer = opt(trainable_params, lr=1.0)
            self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, update_rule)
            
        elif opt_cfg['lr_scheduler'] == 'plateau':
            self.optimizer = opt(trainable_params, lr=opt_cfg['lr'], eps=opt_cfg['eps'])
            self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,'min', patience=1)
            
        else:
            self.lr_scheduler = None
            self.optimizer = opt(trainable_params, lr=opt_cfg['lr'], eps=opt_cfg['eps'])       
            
        self.step = 0 

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
        wavs = [sample[1].transpose(0,1) for sample in samples] 
        features = torch.nn.utils.rnn.pad_sequence(wavs, batch_first=True)
        lengths = torch.tensor([elem.shape[0] for elem in wavs], dtype=torch.int32)
        features = features.view(features.shape[0], features.shape[1]) 
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
        
        ctc_output, encoded_len = self.model(batch.features,
                                             batch.feature_lengths)

        # transpose is BxTxD -> TxBxD 
        total_loss = self.ctc_loss(ctc_output.transpose(
                            0, 1), batch.targets,
                            encoded_len, batch.target_lengths)

        self.log(f"Losses/{step_type}_loss", total_loss, on_step=True, on_epoch=True)        
            
        wer_ctc = cal_er(self.tokenizer, ctc_output, batch.targets, ctc=True)    
        self.log(f"WER/{step_type}_ctc", wer_ctc, on_step=True, on_epoch=True)

        return total_loss

    def configure_optimizers(self):
        return (
            [self.optimizer]#,
           # [
        #        {
         #           "scheduler": self.lr_scheduler,
          #          "monitor": "WER/val_att",
           #         "interval": "epoch",
            #    }
               
           # ],
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
        tr_loader_bs = 1 if self.bucketing and (not self.ascending) else self.config['data']['corpus']['batch_size']
        bucket_size = self.config['data']['corpus']['batch_size'] if self.bucketing and (not self.ascending) else 1  # Ascending without bucketing

        dataset = GigaDataset(self.gigaspeech_path, 
                                'XL_munged',
                                None, # No tokenizer - do in collate
                                bucket_size, 
                                self.ascending) 

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=tr_loader_bs,
            collate_fn=self._train_collate_fn,
            num_workers=self.config['n_jobs'], 
            shuffle=False,
            pin_memory=True
        )
        return dataloader

    def val_dataloader(self):
        # no bucketing in validation
        dataset = GigaDataset(self.gigaspeech_path, 
                                'DEV_munged',
                                None, # No tokenizer - do in collate
                                1, # Batching handled by Trainer
                                True) # Sort in ascending order 

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.config['data']['corpus']['batch_size'],
            collate_fn=self._valid_collate_fn,
            num_workers=self.config['n_jobs'],
        )
        return dataloader

    def test_dataloader(self):
        dataset = torchaudio.datasets.LIBRISPEECH(self.gigaspeech_path, url="test-clean")
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, collate_fn=self._test_collate_fn)
        return dataloader
