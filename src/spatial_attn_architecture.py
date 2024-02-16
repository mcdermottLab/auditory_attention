import torch
import numpy as np
import torch.nn as nn
from collections import OrderedDict
from src.layers import conv2d_same
from src.custom_modules import HannPooling2d

class SimpleAttentionalGain(nn.Module):
    def __init__(self, frequency_dim, cnn_channels, global_avg_cue=False):
        super(SimpleAttentionalGain, self).__init__()
        self.frequency_dim = frequency_dim
        if global_avg_cue:
            self.ouptut_shape = (1,1)
            # self.time_average = nn.AdaptiveAvgPool2d((1, 1)) # outsize is N, C, 1, 1
        else:
            self.ouptut_shape = (frequency_dim, 1)
            # self.time_average = nn.AdaptiveAvgPool2d((frequency_dim, 1)) # outsize is N, C, FreqDim, 1
        self.bias = nn.Parameter(torch.zeros(1)) # init gain scaling to zero
        self.slope = nn.Parameter(torch.ones(1)) # init slope to one
        self.threshold = nn.Parameter(torch.zeros(1)) # init threshold to zero
        self.reset_parameters() 

    def reset_parameters(self):
        nn.init.constant_(self.bias, 0)
        nn.init.constant_(self.slope, 1)
        nn.init.constant_(self.threshold, 0)

    def forward(self, cue, mixture, cue_mask_ixs):
        ## Process cue 
        # time average - same as nn op, but is compat with compile 
        cue = cue.mean(axis=-1,keepdim=True)
        # cue = self.time_average(cue)
        # apply threshold shift
        cue = cue - self.threshold
        # apply slope
        cue = cue * self.slope
        # apply sigmoid & bias
        gain = self.bias + (1-self.bias) * torch.sigmoid(cue)
        ## account for no-cue examples - no gain scaling applied
        if cue_mask_ixs is not None:
            gain[cue_mask_ixs,:] = 1
        # Apply to mixture (element mult)
        mixture = torch.mul(mixture, gain)
        return mixture


class CNN2DExtractor(nn.Module):
    ''' CNN wrapper, includes relu and layer-norm if applied'''

    def __init__(self, input_sr, out_channels, kernel, stride, padding, pool_stride, pool_size, pool_padding, attn, dropout, fc_size=512, global_avg_cue=False, num_classes={"num_words":998, "num_locs":504}, double_size=False, **kwargs):
        super(CNN2DExtractor, self).__init__()
        # Setup
        print(f"{num_classes=}")
        self.dual_task = False
        if isinstance(num_classes, dict):
            class_keys = num_classes.keys()
            if ("num_words" in class_keys) and not ("num_locs" in class_keys):
                # only_word 
                num_classes = num_classes['num_words']
                print('Model performing word task')
            elif ("num_locs" in class_keys) and not ("num_words" in class_keys):
                # only_loc
                num_classes = num_classes['num_locs']
                print('Model performing location task')

            elif ("num_locs" in class_keys) and ("num_words" in class_keys):
                self.dual_task = True
                num_words = num_classes['num_words']
                num_locs = num_classes['num_locs']
                print('Model performing both location and word tasks')

        self.input_sr = input_sr
        self.out_channels = out_channels
        self.kernel = kernel
        self.stride = stride
        self.padding = padding
        self.pool_stride = pool_stride
        self.pool_size = pool_size
        self.pool_padding = pool_padding
        self.attn = attn
        self.frequency_dim = 40
        self.n_layers = len(out_channels)

        self.input_channels = kwargs.get('input_channels', 2)

        self.model_dict = nn.ModuleDict()
        self.output_height = self.frequency_dim
        self.output_len = 20000 # softcode eventually
        self.model_dict["norm_coch_rep"]= nn.LayerNorm([self.input_channels, self.frequency_dim, self.output_len])
        self.model_dict["attn_block_in"] = SimpleAttentionalGain(self.frequency_dim, self.input_channels, global_avg_cue=global_avg_cue)

        for idx in range(self.n_layers):
            nIn = self.input_channels if idx == 0 else out_channels[idx - 1]
            nOut = out_channels[idx]
            # Convolutional block:
            if self.pool_stride[idx] != -1:
                block = nn.Sequential(nn.LayerNorm([nIn, self.output_height, self.output_len]),
                                    conv2d_same.create_conv2d_pad(nIn, nOut, self.kernel[idx], stride=self.stride[idx], padding=self.padding[idx]),
                                    nn.ReLU(),
                                    HannPooling2d(stride=self.pool_stride[idx], pool_size=self.pool_size[idx], padding=self.pool_padding[idx]))
            else:
                block = nn.Sequential(nn.LayerNorm([nIn, self.output_height, self.output_len]),
                                    conv2d_same.create_conv2d_pad(nIn, nOut, self.kernel[idx], stride=self.stride[idx], padding=self.padding[idx]),
                                    nn.ReLU())
            self.model_dict[f'conv_block_{idx}'] = block

            # Compute output shapes using conv formula [(Height - Filter + 2Pad)/ Stride]+1
            if self.padding[idx] == 'same':
                pass
            else:
                self.output_height = int(np.floor((self.output_height - kernel[idx][0] + 2 * padding[idx][0]) / stride[idx][0]) + 1)
                self.output_len = int(np.floor((self.output_len -  kernel[idx][1] + 2 * padding[idx][1]) / stride[idx][1]) + 1)
            if self.pool_stride[idx] != -1:
                # pooling layers
                self.output_height = int(np.floor((self.output_height - pool_size[idx][0] + 2 * pool_padding[idx][0]) / pool_stride[idx][0]) + 1)
                self.output_len = int(np.floor((self.output_len - pool_size[idx][1] + 2 * pool_padding[idx][1]) / pool_stride[idx][1]) + 1)
            # Attentional block:
            if self.attn[idx] == 1:
                self.model_dict[f'attn{idx}'] = SimpleAttentionalGain(self.output_height, nOut, global_avg_cue=global_avg_cue)

        self.output_size = self.output_height * nOut * self.output_len
        self.fullyconnected = nn.Linear(self.output_size, fc_size)
        self.relufc = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        if self.dual_task:
            self.classificationWord = nn.Linear(fc_size, num_words)
            self.classificationLoc = nn.Linear(fc_size, num_locs)
        else:
            self.classification = nn.Linear(fc_size, num_classes)

    def forward(self, cue=None, mixture=None, cue_mask_ixs=None):
        # pass cue through cnn & store reps
        if cue == None:
            mixture = self.model_dict["norm_coch_rep"](mixture)
            for idx in range(self.n_layers):
                mixture = self.model_dict[f'conv_block_{idx}'](mixture)
            out = mixture

        else:
            cue = self.model_dict["norm_coch_rep"](cue)
            mixture = self.model_dict["norm_coch_rep"](mixture)
            attn = self.model_dict["attn_block_in"](cue, mixture, cue_mask_ixs)
            for idx in range(self.n_layers):
                cue = self.model_dict[f'conv_block_{idx}'](cue)
                attn = self.model_dict[f'conv_block_{idx}'](attn)
                if self.attn[idx] == 1:
                    attn = self.model_dict[f'attn{idx}'](cue, attn, cue_mask_ixs)
            out = attn

        out = out.view(out.size(0), self.output_size) # B x FC size
        out = self.fullyconnected(out)        
        out = self.relufc(out)
        out = self.dropout(out)        
        if self.dual_task:
            word_out = self.classificationWord(out)
            loc_out = self.classificationLoc(out)
            return word_out, loc_out
        else:
            return self.classification(out)


class BaseAuditoryNetworkForTransfer(nn.Module):
    ''' CNN wrapper, includes relu and layer-norm if applied'''

    def __init__(self, input_sr, out_channels, kernel, stride, padding, pool_stride, pool_size, pool_padding, attn, n_layers=None, **kwargs):
        super(BaseAuditoryNetworkForTransfer, self).__init__()
        # Setup
        self.input_sr = input_sr
        self.out_channels = out_channels
        self.kernel = kernel
        self.stride = stride
        self.padding = padding
        self.pool_stride = pool_stride
        self.pool_size = pool_size
        self.pool_padding = pool_padding
        self.attn = attn
        self.frequency_dim = 40
        self.n_layers = n_layers if n_layers else len(out_channels)

        self.input_channels = kwargs.get('input_channels', 2)

        self.model_dict = nn.ModuleDict()
        self.output_height = self.frequency_dim
        self.output_len = 20000 # softcode eventually
        self.model_dict["norm_coch_rep"]= nn.LayerNorm([self.input_channels, self.frequency_dim, self.output_len])

        for idx in range(self.n_layers):
            nIn = self.input_channels if idx == 0 else out_channels[idx - 1]
            nOut = out_channels[idx]
            # Convolutional block:
            if self.pool_stride[idx] != -1 and idx != self.n_layers - 1:
                block = nn.Sequential(nn.LayerNorm([nIn, self.output_height, self.output_len]),
                                    conv2d_same.create_conv2d_pad(nIn, nOut, self.kernel[idx], stride=self.stride[idx], padding=self.padding[idx]),
                                    nn.ReLU(),
                                    HannPooling2d(stride=self.pool_stride[idx], pool_size=self.pool_size[idx], padding=self.pool_padding[idx]))
            else:
                block = nn.Sequential(nn.LayerNorm([nIn, self.output_height, self.output_len]),
                                    conv2d_same.create_conv2d_pad(nIn, nOut, self.kernel[idx], stride=self.stride[idx], padding=self.padding[idx]),
                                    nn.ReLU())
            self.model_dict[f'conv_block_{idx}'] = block

            # Compute output shapes using conv formula [(Height - Filter + 2Pad)/ Stride]+1
            if self.padding[idx] == 'same':
                pass
            else:
                self.output_height = int(np.floor((self.output_height - kernel[idx][0] + 2 * padding[idx][0]) / stride[idx][0]) + 1)
                self.output_len = int(np.floor((self.output_len -  kernel[idx][1] + 2 * padding[idx][1]) / stride[idx][1]) + 1)
            if self.pool_stride[idx] != -1:
                # pooling layers
                self.output_height = int(np.floor((self.output_height - pool_size[idx][0] + 2 * pool_padding[idx][0]) / pool_stride[idx][0]) + 1)
                self.output_len = int(np.floor((self.output_len - pool_size[idx][1] + 2 * pool_padding[idx][1]) / pool_stride[idx][1]) + 1)

        self.output_size = self.output_height * nOut * self.output_len

    def forward(self, x):
        # pass cue through cnn & store reps
        x = self.model_dict["norm_coch_rep"](x)
        for idx in range(self.n_layers):
            x = self.model_dict[f'conv_block_{idx}'](x)
        x = x.view(x.size(0), self.output_size) # B x FC size
        return x 
