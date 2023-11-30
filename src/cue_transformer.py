import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Any, Union, Callable


# Make 2D CNN to downsample inputs from 8kHz to 50Hz - similar to wav2vec2 feature encoder but 2d instead of 1d
class FeatureEncoder(nn.Module):
    def __init__(self):
        ## 7 layer cnn with 1 input channel and 128 output channels, strides of (5,2,2,2,2,2,2) and 2d kernels of size (10,3,3,3,3,2,2) with layeer norm and relu activation
        # expects cochleagram inputs 
        # TODO: try without cochlagram inputs
        super(FeatureEncoder, self).__init__()
        self.conv = nn.Sequential(
            nn.LayerNorm([1, 40, 16000]),
            nn.Conv2d(1, 128, kernel_size=(2, 10), stride=(2, 5), bias=False),
            nn.ReLU(inplace=True),
            nn.LayerNorm([128, 20, 3200]),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2), bias=False),
            nn.ReLU(inplace=True),
            nn.LayerNorm([128, 10, 1600]),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2), bias=False),
            nn.ReLU(inplace=True),
            nn.LayerNorm([128, 5, 800]),    
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 2), bias=False), 
            nn.ReLU(inplace=True),
            nn.LayerNorm([128, 3, 400]),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 2), bias=False),
            nn.ReLU(inplace=True),
            nn.LayerNorm([128, 2, 200]),
            nn.Conv2d(128, 128, kernel_size=(2, 2), stride=(1, 2), bias=False),
            nn.ReLU(inplace=True)
        )
        self.norm = nn.LayerNorm([128, 1, 100])
        self.linear = nn.Linear(128, 128)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        # reshape to 2d
        x = x.view(x.shape[0], -1)
        x = self.linear(x)
        x = self.dropout(x)
        return x

# 7 layer 1d cnn with 1 input channel and 128 output channels, strides of (5,2,2,2,2,2,2) and 2d kernels of size (10,3,3,3,3,2,2) with layer norm and relu activation
class FeatureEncoder1D(nn.Module):
    # expects 1d 2 second waveform inputs at 20kHz output at 50Hz
    def __init__(self):
        super(FeatureEncoder1D, self).__init__()
        self.conv = nn.Sequential(
            nn.LayerNorm([1, 40000]),
            nn.Conv1d(1, 512, kernel_size=10, stride=5, bias=False),
            nn.ReLU(inplace=True),
            nn.LayerNorm([512, 7998]),
            nn.Conv1d(512, 512, kernel_size=3, stride=2, bias=False),
            nn.ReLU(inplace=True),
            nn.LayerNorm([512, 3999]),
            nn.Conv1d(512, 512, kernel_size=3, stride=2, bias=False),
            nn.ReLU(inplace=True),
            nn.LayerNorm([512, 2000]),
            nn.Conv1d(512, 512, kernel_size=3, stride=2, bias=False),
            nn.ReLU(inplace=True),
            nn.LayerNorm([512, 1000]),
            nn.Conv1d(512, 512, kernel_size=3, stride=2, bias=False),
            nn.ReLU(inplace=True),
            nn.LayerNorm([512, 500]),
            nn.Conv1d(512, 512, kernel_size=3, stride=2, bias=False),
            nn.ReLU(inplace=True),
            nn.LayerNorm([512, 250]),
            nn.Conv1d(512, 512, kernel_size=3, stride=2, bias=False),
            nn.ReLU(inplace=True)
        )
        self.norm = nn.LayerNorm([512, 125]),
        self.linear = nn.Linear(512, 128)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


# Convolutional positional encoding
class ConvPositionalEncoding(nn.Module):
    def __init__(self):
        super(ConvPositionalEncoding, self).__init__()
        self.conv = nn.Conv1d(128, 128, kernel_size=1, groups=16, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.norm = nn.LayerNorm(128, eps=1e-5)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x_pos = self.relu(self.conv(x))
        x = x + x_pos
        x = self.norm(x)
        x = self.dropout(x)
        return x
    
class CueTransformerLayer(nn.TransformerDecoderLayer):
    def __init__(self,
                 d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                 bias: bool = True, device=None, dtype=None) -> None:
        super(CueTransformerLayer, self).__init__(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout,
                 activation=activation, layer_norm_eps=layer_norm_eps, batch_first=batch_first, norm_first=norm_first,
                 bias=bias, device=device, dtype=dtype)
    
    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        tgt_is_causal: bool = False,
        memory_is_causal: bool = False,
    ) -> Tensor:
        r"""Pass the inputs (and mask) through the decoder layer.
        For mixture (tgt) and cue (memory) are both passed through 
        self attention. Cross attention is applied between the cue and mixture.

        Args:
            tgt: the mixture sequence to the decoder layer (required).
            memory: the cue sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).
            tgt_is_causal: If specified, applies a causal mask as ``tgt mask``.
                Default: ``False``.
                Warning:
                ``tgt_is_causal`` provides a hint that ``tgt_mask`` is
                the causal mask. Providing incorrect hints can result in
                incorrect execution, including forward and backward
                compatibility.
            memory_is_causal: If specified, applies a causal mask as
                ``memory mask``.
                Default: ``False``.
                Warning:
                ``memory_is_causal`` provides a hint that
                ``memory_mask`` is the causal mask. Providing incorrect
                hints can result in incorrect execution, including
                forward and backward compatibility.

        Shape:
            see the docs in Transformer class.
        """
        x = tgt
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), tgt_mask, tgt_key_padding_mask, tgt_is_causal)
            memory = memory + self._sa_block(self.norm1(memory), memory_mask, memory_key_padding_mask, memory_is_causal)
            x = x + self._mha_block(self.norm2(x), memory, memory_mask, memory_key_padding_mask, memory_is_causal)
            x = x + self._ff_block(self.norm3(x))
            memory = memory + self._ff_block(self.norm3(memory))
        else:
            x = self.norm1(x + self._sa_block(x, tgt_mask, tgt_key_padding_mask, tgt_is_causal))
            memory = self.norm1(memory + self._sa_block(memory, memory_mask, memory_key_padding_mask, memory_is_causal))
            x = self.norm2(x + self._mha_block(x, memory, memory_mask, memory_key_padding_mask, memory_is_causal))
            x = self.norm3(x + self._ff_block(x))
            memory = self.norm3(memory + self._ff_block(memory))

        return x, memory

# Use torch transformer decoder that operates on two inputs, a cue and mixture spectrogram, and uses cross attention between the cue and mixtuer to predict words said in the mixture by the cue speaker.
class TransformerCueModel(torch.nn.Module):
    def __init__(self,
                 num_classes=800,
                 d_model=128,
                 nhead=8,
                 dim_feedforward=128,
                 dropout=0.1,
                 batch_first=True,
                 activation='relu',
                 bias=False,
                 num_layers=6,
                 tgt_is_causal=False,
                 memory_is_causal=False):
        super(TransformerCueModel, self).__init__()
        # init decoder layer template
        self.tgt_is_causal = tgt_is_causal
        self.memory_is_causal = memory_is_causal

        # create stack of num_layers CueTransformerLayer layers as module dict
        self.transformer = nn.ModuleDict({f"transformer_block_{ix}" : CueTransformerLayer(d_model=d_model,
                                            nhead=nhead,
                                            dim_feedforward=dim_feedforward,
                                            dropout=dropout,
                                            batch_first=batch_first,
                                            activation=activation,
                                            bias=bias) for ix in range(num_layers)})
        # init model 
        self.feature_encoder = FeatureEncoder1D()
        self.positional_encoder = ConvPositionalEncoding()
        # self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, cue, mixture):
        # encode cue and mixture waveforms 
        cue = self.feature_encoder(cue)
        mixture = self.feature_encoder(mixture)
        # positional encoding
        cue = self.positional_encoder(cue)
        mixture = self.positional_encoder(mixture)
        # transformer
        for layer in self.transformer.values():
            mixture, cue = layer(tgt=mixture,
                                 memory=cue,
                                 tgt_is_causal=self.tgt_is_causal,
                                 memory_is_causal=self.memory_is_causal)
        # classifier
        mixture = self.classifier(mixture)
        return mixture

