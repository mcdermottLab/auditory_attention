import torch
import numpy as np
import torch.nn as nn
from collections import OrderedDict
from src.layers import conv2d_same
from src.custom_modules import HannPooling2d


# Calculate symmetric padding for a convolution
def get_padding(kernel_size: int, stride: int = 1, dilation: int = 1, **_) -> int:
    if isinstance(kernel_size, list):
        h_pad = ((stride - 1) + dilation * (kernel_size[0] - 1)) // 2
        w_pad = ((stride - 1) + dilation * (kernel_size[1] - 1)) // 2
        return [h_pad, w_pad]

    padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
    return padding


class SimpleAttentionalGain(nn.Module):
    def __init__(self, frequency_dim, cnn_channels, global_avg_cue=False):
        super(SimpleAttentionalGain, self).__init__()
        if global_avg_cue:
            self.time_average = nn.AdaptiveAvgPool2d((1, 1)) # outsize is N, C, 1, 1
        else:
            self.time_average = nn.AdaptiveAvgPool2d((frequency_dim, 1)) # outsize is N, C, FreqDim, 1
        self.bias = nn.Parameter(torch.zeros(1)) # init gain scaling to zero
        self.slope = nn.Parameter(torch.ones(1)) # init slope to one
        self.threshold = nn.Parameter(torch.zeros(1)) # init threshold to zero
        self.reset_parameters() 

    def reset_parameters(self):
        nn.init.constant_(self.bias, 0)
        nn.init.constant_(self.slope, 1)
        nn.init.constant_(self.threshold, 0)

    def forward(self, cue, mixture, cue_mask_ixs=None):
        ## Process cue 
        cue = self.time_average(cue)
        # apply threshold shift
        cue = cue - self.threshold
        # apply slope
        cue = cue * self.slope
        # apply sigmoid & bias
        cue = self.bias + (1-self.bias) * torch.sigmoid(cue)
        ## account for no-cue examples - no gain scaling applied
        if cue_mask_ixs is not None:
            gain[cue_mask_ixs,:] = 1 
        # Apply to mixture (element mult)
        mixture = torch.mul(mixture, cue)
        return mixture


class KernelAttentionalGain(nn.Module):
    def __init__(self, frequency_dim, cnn_channels, global_avg_cue=False):
        super(SimpleAttentionalGain, self).__init__()
        if global_avg_cue:
            self.time_average = nn.AdaptiveAvgPool2d((1, 1)) # outsize is N, C, 1, 1
        else:
            self.time_average = nn.AdaptiveAvgPool2d((frequency_dim, 1)) # outsize is N, C, FreqDim, 1
        self.bias = nn.Parameter(torch.zeros(1, cnn_channels, 1, 1)) # init gain scaling to zero
        self.slope = nn.Parameter(torch.ones(1, cnn_channels, 1, 1)) # init slope to one
        self.threshold = nn.Parameter(torch.zeros(1, cnn_channels, 1, 1)) # init threshold to zero
        self.reset_parameters() 

    def reset_parameters(self):
        nn.init.constant_(self.bias, 0)
        nn.init.constant_(self.slope, 1)
        nn.init.constant_(self.threshold, 0)

    def forward(self, cue, mixture):
        ## Process cue 
        cue = self.time_average(cue)
        # apply threshold shift
        cue = cue - self.threshold
        # apply slope
        cue = cue * self.slope
        # apply sigmoid & bias
        cue = self.bias + (1-self.bias) * torch.sigmoid(cue)
        # Apply to mixture (element mult)
        mixture = torch.mul(mixture, cue)
        return mixture


class SeparableConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, bias=False, padding=1, **kwargs):
        super(SeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, 
                                groups=in_channels, bias=bias, padding=padding)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 
                                kernel_size=1, bias=bias, padding=1)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out



class DynamicCNN(nn.Module):
    ''' CNN wrapper, includes relu and layer-norm if applied'''
    def __init__(self, input_c, input_h, input_w, norm, out_channels, kernel, stride, padding, pool, pool_stride, pool_size, attn, dropout,
                 fc_size=512, global_avg_cue=False, num_classes={"num_words":998, "num_locs":504}, conv_type='tf_compat', act_type='ReLU',
                 standard_classifier=True, global_avg_pool=False, **kwargs):
        super(DynamicCNN, self).__init__()
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

        self.n_layers = len(out_channels)
        self.input_c = input_c
        self.out_channels = out_channels
        self.kernel = kernel
        self.stride = stride
        self.padding = padding
        self.pool_stride = pool_stride
        self.pool_size = pool_size
        self.attn = attn
        self.frequency_dim = input_h
        pool_padding = [[0,0]] * self.n_layers
        self.pool_padding = pool_padding

        self.global_avg_pool = global_avg_pool
        self.standard_classifier = standard_classifier


        self.model_dict = nn.ModuleDict()
        self.output_height = self.frequency_dim
        self.output_len = input_w 
        self.model_dict["norm_coch_rep"]= nn.LayerNorm([self.input_c, self.frequency_dim, self.output_len])
        self.model_dict["attn_block_in"] = SimpleAttentionalGain(self.frequency_dim, self.input_c, global_avg_cue=global_avg_cue)

        if conv_type == 'SeparableConv2d':
            self.conv = SeparableConv2d
        elif conv_type == 'tf_compat':
            self.conv = conv2d_same.create_conv2d_pad
        else:
            self.conv = nn.Conv2d

        self.act = nn.GELU if act_type == 'GELU' else nn.ReLU
    
        for idx in range(self.n_layers):
            nIn = self.input_c if idx == 0 else out_channels[idx - 1]
            nOut = out_channels[idx]
            # Convolutional block:
            conv_layers = []
            if norm[idx] == 1:
                conv_layers.append(nn.LayerNorm([nIn, self.output_height, self.output_len]))


            # Compute output shapes using conv formula [(Height - Filter + 2Pad)/ Stride]+1
            if self.padding[idx] == 'same':
                padding[idx] = get_padding(kernel[idx])
            elif self.padding[idx] == 'valid':
                padding[idx] = [0, 0]

            conv_layers.append(self.conv(nIn, nOut, self.kernel[idx], stride=self.stride[idx], padding=self.padding[idx]))
            conv_layers.append(self.act())

            if conv_type == 'SeparableConv2d':
                # depthwise - padding = 1 
                self.output_height = int(np.floor((self.output_height - kernel[idx][0] + 2 * padding[idx][0]) / stride[idx][0]) + 1)
                self.output_len = int(np.floor((self.output_len -  kernel[idx][1] + 2 * padding[idx][1]) / stride[idx][1]) + 1)
                # pointwise - kernel = 1 padding = 1
                self.output_height = int(np.floor((self.output_height - 1 + 2) / 1) + 1)
                self.output_len = int(np.floor((self.output_len -  1 + 2 ) / 1) + 1)
            else:
                self.output_height = int(np.floor((self.output_height - kernel[idx][0] + 2 * padding[idx][0]) / stride[idx][0]) + 1)
                self.output_len = int(np.floor((self.output_len -  kernel[idx][1] + 2 * padding[idx][1]) / stride[idx][1]) + 1)

            # pooling layers
            if pool[idx] == 1:
                # add layer
                conv_layers.append(HannPooling2d(stride=self.pool_stride[idx], pool_size=self.pool_size[idx], padding=self.pool_padding[idx]))    
                # update shape 
                self.output_height = int(np.floor((self.output_height - pool_size[idx][0] + 2 * pool_padding[idx][0]) / pool_stride[idx][0]) + 1)
                self.output_len = int(np.floor((self.output_len - pool_size[idx][1] + 2 * pool_padding[idx][1]) / pool_stride[idx][1]) + 1)

            self.model_dict[f'conv_block_{idx}'] = nn.Sequential(*conv_layers)
            # Attentional block:
            if self.attn[idx] == 1:
                self.model_dict[f'attn{idx}'] = SimpleAttentionalGain(self.output_height, nOut, global_avg_cue=global_avg_cue)
        
        print(f"{self.output_height=}")
        print(f"{self.output_len=}")
        print(f"{self.n_layers=}")
        if self.standard_classifier:
            self.output_size =  nOut * self.output_height * self.output_len 
            self.fullyconnected = nn.Linear(self.output_size, fc_size)
            self.relufc = self.act()
            self.dropout = nn.Dropout(dropout)

        elif self.global_avg_pool:
            self.output_size = nOut
            fc_size = nOut
            self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        

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

        if self.standard_classifier:
            out = out.view(out.size(0), self.output_size) # B x FC size
            out = self.fullyconnected(out)        
            out = self.relufc(out)
            out = self.dropout(out)   
        
        elif self.global_avg_pool:
            out = self.global_avg_pool(out)
            out = torch.flatten(out, 1)

        if self.dual_task:
            word_out = self.classificationWord(out)
            loc_out = self.classificationLoc(out)
            return word_out, loc_out
        else:
            return self.classification(out)