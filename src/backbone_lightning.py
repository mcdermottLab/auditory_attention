import torch
import tensorflow as tf
from pathlib import Path

from pytorch_lightning import LightningModule
from torchmetrics.classification import Accuracy

import src.custom_modules as cm
import src.util_tfrecord as util_tfrecords
from src.spatial_attn_architecture import BackBoneCNN

class BinauralBackBoneModule(LightningModule):
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

        self.corpora_name = config.get('corpora_name', False)

        num_words = self.model_config['num_classes']['num_words']

        # Init Model
        # Get model architecture
        self.model = BackBoneCNN(**self.model_config)
        # self.model = torch.compile(model, mode="default")

        # Add input rep to model or audio transforms
        self.rep_on_gpu = self.audio_config['rep_kwargs']['rep_on_gpu']
        self.coch_gram = cm.AttnAudioInputRepresentation(**self.audio_config).to(self.device)

        # Losses
        self.word_loss_fn = torch.nn.CrossEntropyLoss(ignore_index=0)

        self.train_acc = torch.nn.ModuleDict({'word':Accuracy(task="multiclass", num_classes=num_words)})
        self.valid_acc = torch.nn.ModuleDict({'word':Accuracy(task="multiclass", num_classes=num_words)})
        self.test_acc = torch.nn.ModuleDict({'word':Accuracy(task="multiclass", num_classes=num_words)})

        self.test_confusion = torch.nn.ModuleDict({'word': Accuracy(task="multiclass", num_classes=num_words)})
        self.accuracy = {'train': self.train_acc,
                         'val': self.valid_acc,
                         'test': self.test_acc,
                         'test_confusion': self.test_confusion
                        }

        # Optimizer
        opt = getattr(torch.optim, self.hparas_config['optimizer'])
        model_params = [{'params': self.model.parameters()}]
        self.optimizer = opt(model_params, lr=self.hparas_config['lr'], eps=self.hparas_config['eps'])    

    def _step(self, batch, batch_idx, step_type):
        if batch is None:
            return None


        sound = torch.from_numpy(batch['signal'].numpy())
        exs, time, channels = sound.shape
        sound = sound.view(exs, channels, 1, time).to(self.device)

        # process labels 
        word_label = torch.from_numpy(batch['word_int'].numpy()).type(torch.LongTensor).squeeze().to(self.device)

        # features to cochleagram
        sound, _ = self.coch_gram(sound, None)
        # self() is self.forward()
        word_pred = self(sound)

        # filter valid examples for task losses
        w_idx = torch.argwhere(word_label > 0).squeeze()
        # word screen
        word_pred = word_pred.index_select(0, w_idx)
        word_label = word_label.index_select(0, w_idx)


        # calc losses
        if w_idx.numel() > 0:
            word_loss = self.word_loss_fn(word_pred, word_label)
            self.accuracy[step_type]['word'](word_pred, word_label) # word accuracy
            self.log(f"{step_type}_word_acc", self.accuracy[step_type]['word'], on_step=False, on_epoch=True, prog_bar=True)
            self.log(f"{step_type}_word_loss", word_loss.detach(), on_step=True, on_epoch=False, prog_bar=True)

        else:
            word_loss = torch.tensor(0, device=self.device, dtype=torch.float32) #  
            
        loss = word_loss 
        self.log(f"{step_type}_loss", loss.detach(), on_step=True, on_epoch=False, prog_bar=True)

        return loss

    def configure_optimizers(self):
        return [self.optimizer]

    def forward(self, sound: torch.tensor):
        outputs = self.model(mixture=sound)
        # Outputs here are logits
        return outputs

    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "val")

    def train_dataloader(self):
        filenames = [path.as_posix() for path in Path(self.corpora_config["root"]).glob("train/*.tfrecords")]
        dataset = util_tfrecords.get_dataset_from_tfrecords(
                filenames,
                feature_description='config_feature_description.pkl',
                features_to_exclude=['list_'],
                bytes_description='config_bytes_description.pkl',
                compression_type='GZIP',
                eval_mode=False,
                buffer_size_prefetch=tf.data.AUTOTUNE,
                buffer_size_shuffle=100,
                batch_size=self.hparas_config['batch_size'],
                shuffle_seed=None,
                densify_downsample_factors=None,
                densify_jitter_indices=None,
                densify_dtype=tf.float32,
                filter_function=None,
                map_function=None,
                verbose=False)
        return dataset

    def val_dataloader(self):
        filenames = [path.as_posix() for path in Path(self.corpora_config["root"]).glob("valid/*.tfrecords")]
        dataset = util_tfrecords.get_dataset_from_tfrecords(
                filenames,
                feature_description='config_feature_description.pkl',
                features_to_exclude=['list_'],
                bytes_description='config_bytes_description.pkl',
                compression_type='GZIP',
                eval_mode=True,
                buffer_size_prefetch=tf.data.AUTOTUNE,
                buffer_size_shuffle=100,
                batch_size=self.hparas_config['batch_size'],
                shuffle_seed=None,
                densify_downsample_factors=None,
                densify_jitter_indices=None,
                densify_dtype=tf.float32,
                filter_function=None,
                map_function=None,
                verbose=False)
        return dataset



