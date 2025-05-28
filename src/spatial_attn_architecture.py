import torch
import os 
import re 
import sys
import json 
import numpy as np
import torch.nn as nn
from collections import OrderedDict
from src.layers import conv2d_same
from src.layers import padding as pad_utils
from src.custom_modules import HannPooling2d

class SimpleAttentionalGain(nn.Module):
    def __init__(self, frequency_dim, cnn_channels, additive=False, time_dim=-1, **kwargs):
        super(SimpleAttentionalGain, self).__init__()
        self.frequency_dim = frequency_dim
        self.time_dim = time_dim
        self.bias = nn.Parameter(torch.zeros(1)) # init gain scaling to zero
        self.slope = nn.Parameter(torch.ones(1)) # init slope to one
        self.threshold = nn.Parameter(torch.zeros(1)) # init threshold to zero
        self.additive = additive # if True, gain is added to mixture, else multiplied
        self.reset_parameters() 

    def reset_parameters(self):
        nn.init.constant_(self.bias, 0)
        nn.init.constant_(self.slope, 1)
        nn.init.constant_(self.threshold, 0)

    def forward(self, cue, mixture, cue_mask_ixs):
        ## Process cue 
        # time average - same as nn op, but is compat with compile 
        cue = cue.mean(axis=self.time_dim,keepdim=True)
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
        if self.additive:
            mixture = torch.add(mixture, gain)
        else:
            mixture = torch.mul(mixture, gain)
        return mixture
    


class BinauralAuditoryAttentionCNN(nn.Module):
    def __init__(self, input_sr, out_channels, kernel, stride, padding, pool_stride, pool_size, pool_padding, attn, dropout,
                  fc_size=512, global_avg_cue=False, num_classes={"num_words":800, "num_locs":504}, frequency_dim=40,
                  residual_attn=False, n_cue_frames=None, starting_output_len = 20000, norm_first=True, ln_affine=True, 
                  v08=False, additive=False, cue_loc_task=False, fc_attn=True,  per_kernel_gain=False, **kwargs):
        super(BinauralAuditoryAttentionCNN, self).__init__()
        # Setup
        # print(f"{num_classes=}")
        self.dual_task = False
        if isinstance(num_classes, dict):
            class_keys = num_classes.keys()
            if ("num_words" in class_keys) and not ("num_locs" in class_keys):
                # only_word 
                self.num_classes = num_classes['num_words']
                # print('Model performing word task')
            elif ("num_locs" in class_keys) and not ("num_words" in class_keys):
                # only_loc
                self.num_classes = num_classes['num_locs']
                # print('Model performing location task')

            elif ("num_locs" in class_keys) and ("num_words" in class_keys):
                self.dual_task = True
                self.num_words = num_classes['num_words']
                self.num_locs = num_classes['num_locs']
                # print('Model performing both location and word tasks')

        self.input_sr = input_sr
        self.out_channels = out_channels
        self.kernel = kernel
        self.stride = stride
        self.padding = padding
        self.pool_stride = pool_stride
        self.pool_size = pool_size
        self.pool_padding = pool_padding
        self.attn = attn
        self.frequency_dim = frequency_dim
        self.n_layers = len(out_channels)
        self.norm_first = norm_first
        self.cue_loc_task  = cue_loc_task
        self.per_kernel_gain = per_kernel_gain
        self.gain_module = SimpleAttentionalGain

        # if norm_first:
            # print(f"Conv block order: LN -> Conv -> ReLU")
        # elif not norm_first:
            # print(f"Conv block order: Conv -> ReLU -> LN")

        self.input_channels = kwargs.get('input_channels', 2)
        self.n_cue_frames = n_cue_frames
        self.residual_attn = residual_attn
        self.ln_affine = ln_affine
        if residual_attn:
            print(f"Using residual attention")
        self.v08 = v08
        self.fc_attn = fc_attn
        # print(f"fc_attn: {fc_attn}")

        self.model_dict = nn.ModuleDict()
        self.output_height = frequency_dim
        self.output_len = starting_output_len 

        # build architecture
        if v08:
            coch_affine = self.ln_affine
        else:
            coch_affine = True
        # print('coch_affine:', coch_affine)
        self.model_dict["norm_coch_rep"]= nn.LayerNorm([self.input_channels, self.frequency_dim, self.output_len], elementwise_affine=coch_affine)

        for idx in range(self.n_layers):
            nIn = self.input_channels if idx == 0 else out_channels[idx - 1]
            nOut = out_channels[idx]

            # Attentional block:
            if self.attn[idx] == 1:
                self.model_dict[f'attn{idx}'] = self.gain_module (self.output_height, nIn, global_avg_cue=global_avg_cue, n_cue_frames=self.n_cue_frames, additive=additive)

            # pre-compute conv output sizes - will assign to self.output_height and self.output_len after defining block
            # Sizes will be used for normalization layers, but depend on order of norm and conv 
            # norm -> conv gets prior output shapes, conv -> norm gets new output shapes)
            # compute output shapes using conv formula [(Height + 2Pad - dilation * (kernel - 1) -1) /  Stride] + 1
            # ignoring dilation since it's not used in this model (dilation = 1)
            if self.padding[idx] == 'same':
                output_height = self.output_height
                output_len = self.output_len
            else:
                conv_padding, _ = pad_utils.get_padding_value(self.padding[idx], self.kernel[idx], stride=self.stride[idx])
                output_height = int(np.floor((self.output_height + (2 * conv_padding[0]) - (kernel[idx][0] - 1) - 1) / stride[idx][0]) + 1)
                output_len = int(np.floor((self.output_len + (2 * conv_padding[1]) -  (kernel[idx][1] - 1) - 1) / stride[idx][1]) + 1)
                if self.n_cue_frames:
                    self.n_cue_frames = int(np.floor((self.n_cue_frames + (2 * conv_padding[1]) -  (kernel[idx][1] - 1) - 1) / stride[idx][1]) + 1)

            if self.norm_first:
                # print(f'nIn: {nIn}, nOut: {nOut}, kernel: {self.kernel[idx]}, stride: {self.stride[idx]}, padding: {self.padding[idx]}')
                # print(f"output height: {self.output_height}, output len: {self.output_len}")
                # if norm before conv, can use prior output shapes for norm layer
                block = nn.Sequential(nn.LayerNorm([nIn, self.output_height, self.output_len], elementwise_affine=self.ln_affine),
                                    conv2d_same.create_conv2d_pad(nIn, nOut, self.kernel[idx], stride=self.stride[idx], padding=self.padding[idx]),
                                    nn.ReLU())
            else:  
                # print(f'nIn: {nIn}, nOut: {nOut}, kernel: {self.kernel[idx]}, stride: {self.stride[idx]}, padding: {self.padding[idx]}')
                # print(f"output height: {output_height}, output len: {output_len}")
                # if norm after conv, use new output shapes for norm layer
                block = nn.Sequential(conv2d_same.create_conv2d_pad(nIn, nOut, self.kernel[idx], stride=self.stride[idx], padding=self.padding[idx]),
                                    nn.ReLU(),
                                    nn.LayerNorm([nOut, output_height, output_len], elementwise_affine=self.ln_affine))
            # update post-conv init
            self.output_height, self.output_len = output_height, output_len
            self.model_dict[f'conv_block_{idx}'] = block

            if self.pool_stride[idx] != -1:
                self.model_dict[f'hann_pool_{idx}'] = HannPooling2d(stride=self.pool_stride[idx], pool_size=self.pool_size[idx], padding=self.pool_padding[idx])
                # Compute output shapes for pooling layers using conv formula [(Height - Filter + 2Pad)/ Stride]+1
                self.output_height = int(np.floor((self.output_height - pool_size[idx][0] + 2 * pool_padding[idx][0]) / pool_stride[idx][0]) + 1)
                self.output_len = int(np.floor((self.output_len - pool_size[idx][1] + 2 * pool_padding[idx][1]) / pool_stride[idx][1]) + 1)
                if self.n_cue_frames:
                    self.n_cue_frames = int(np.floor((self.n_cue_frames - pool_size[idx][1] + 2 * pool_padding[idx][1]) / pool_stride[idx][1]) + 1)

        if v08 and fc_attn:
            self.model_dict[f'attnfc'] = self.gain_module (self.output_height, nOut, global_avg_cue=global_avg_cue, n_cue_frames=self.n_cue_frames, additive=additive)

        self.output_size = self.output_height * nOut * self.output_len
        self.fullyconnected = nn.Linear(self.output_size, fc_size)
        self.relufc = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
        if self.cue_loc_task:
            self.loc_fc = nn.Linear(self.output_size, fc_size)
            self.locrelufc = nn.ReLU()
            self.locdropout = nn.Dropout(dropout)

        if self.dual_task or self.cue_loc_task:
            self.classificationWord = nn.Linear(fc_size, self.num_words)
            self.classificationLoc = nn.Linear(fc_size, self.num_locs)
        else:
            self.classification = nn.Linear(fc_size, self.num_classes)
    
    def forward(self, cue=None, mixture=None, cue_mask_ixs=None):
        if cue == None:
            mixture = self.model_dict["norm_coch_rep"](mixture)
            for idx in range(self.n_layers):
                mixture = self.model_dict[f'conv_block_{idx}'](mixture)
                # print(f"conv_block_{idx}, {mixture.mean().item(), mixture.max().item(), mixture.min().item()}")
                if self.pool_stride[idx] != -1:
                    mixture = self.model_dict[f'hann_pool_{idx}'](mixture)
            out = mixture

        else:
            cue = self.model_dict["norm_coch_rep"](cue)
            attn = self.model_dict["norm_coch_rep"](mixture)
            for idx in range(self.n_layers):
                if self.attn[idx] == 1:
                    if self.residual_attn:
                        attn = self.model_dict[f'attn{idx}'](cue, attn, cue_mask_ixs) + attn
                    else:
                        attn = self.model_dict[f'attn{idx}'](cue, attn, cue_mask_ixs)
                        
                # print(f"conv_block_{idx} post gain mean max and min: {attn.mean().item(), attn.max().item(), attn.min().item()}")

                cue = self.model_dict[f'conv_block_{idx}'](cue)
                attn = self.model_dict[f'conv_block_{idx}'](attn)
                # print(f"conv_block_{idx} post conv and norm mean max and min: {attn.mean().item(), attn.max().item(), attn.min().item()}")
                if self.pool_stride[idx] != -1:
                    cue = self.model_dict[f'hann_pool_{idx}'](cue)
                    attn = self.model_dict[f'hann_pool_{idx}'](attn)

            if self.v08 and 'attnfc' in self.model_dict.keys(): 
                attn = self.model_dict['attnfc'](cue, attn, cue_mask_ixs)
            out = attn
            
        out = out.view(out.size(0), self.output_size) # B x FC size
        out = self.fullyconnected(out)        
        out = self.relufc(out)
        out = self.dropout(out)  
        if self.cue_loc_task:
            word_out = self.classificationWord(out)
            cue = cue.view(cue.size(0), self.output_size) # B x FC size
            loc_out = self.loc_fc(cue)
            loc_out = self.locrelufc(loc_out)
            loc_out = self.locdropout(loc_out)
            loc_out = self.classificationLoc(loc_out)
            return word_out, loc_out
            
        if self.dual_task:
            word_out = self.classificationWord(out)
            loc_out = self.classificationLoc(out)
            return word_out, loc_out
        else:
            return self.classification(out)


class BinauralControlCNN(nn.Module):
    def __init__(self, input_sr, out_channels, kernel, stride, padding, pool_stride, pool_size, pool_padding, attn, dropout,
                  fc_size=512, num_classes={"num_words":800, "num_locs":504}, frequency_dim=40,
                  starting_output_len = 20000, norm_first=True, ln_affine=True, v08=False, **kwargs):
        super(BinauralControlCNN, self).__init__()
        # Setup
        print(f"{num_classes=}")
        self.dual_task = False
        if isinstance(num_classes, dict):
            class_keys = num_classes.keys()
            if ("num_words" in class_keys) and not ("num_locs" in class_keys):
                # only_word 
                self.num_classes = num_classes['num_words']
                print('Model performing word task')
            elif ("num_locs" in class_keys) and not ("num_words" in class_keys):
                # only_loc
                self.num_classes = num_classes['num_locs']
                print('Model performing location task')

            elif ("num_locs" in class_keys) and ("num_words" in class_keys):
                self.dual_task = True
                self.num_words = num_classes['num_words']
                self.num_locs = num_classes['num_locs']
                print('Model performing both location and word tasks')

        self.input_sr = input_sr
        self.out_channels = out_channels
        self.kernel = kernel
        self.stride = stride
        self.padding = padding
        self.pool_stride = pool_stride
        self.pool_size = pool_size
        self.pool_padding = pool_padding
        self.frequency_dim = frequency_dim
        self.n_layers = len(out_channels)
        self.norm_first = norm_first
        if norm_first:
            print(f"Conv block order: LN -> Conv -> ReLU")
        elif not norm_first:
            print(f"Conv block order: Conv -> ReLU -> LN")

        self.input_channels = kwargs.get('input_channels', 4)
        self.ln_affine = ln_affine
        self.v08 = v08

        self.model_dict = nn.ModuleDict()
        self.output_height = frequency_dim
        self.output_len = starting_output_len # softcode eventually

        # build architecture
        if v08:
            coch_affine = self.ln_affine
        else:
            coch_affine = True
        self.model_dict["norm_coch_rep"]= nn.LayerNorm([self.input_channels, self.frequency_dim, self.output_len], elementwise_affine=coch_affine)

        for idx in range(self.n_layers):
            nIn = self.input_channels if idx == 0 else out_channels[idx - 1]
            nOut = out_channels[idx]
            # pre-compute conv output sizes - will assign to self.output_height and self.output_len after defining block
            # Sizes will be used for normalization layers, but depend on order of norm and conv 
            # norm -> conv gets prior output shapes, conv -> norm gets new output shapes)
            # compute output shapes using conv formula [(Height + 2Pad - dilation * (kernel - 1) -1) /  Stride] + 1
            # ignoring dilation since it's not used in this model (dilation = 1)
            if self.padding[idx] == 'same':
                output_height = self.output_height
                output_len = self.output_len
            else:
                conv_padding, _ = pad_utils.get_padding_value(self.padding[idx], self.kernel[idx], stride=self.stride[idx])
                output_height = int(np.floor((self.output_height + (2 * conv_padding[0]) - (kernel[idx][0] - 1) - 1) / stride[idx][0]) + 1)
                output_len = int(np.floor((self.output_len + (2 * conv_padding[1]) -  (kernel[idx][1] - 1) - 1) / stride[idx][1]) + 1)
    
            if self.norm_first:
                # print(f'nIn: {nIn}, nOut: {nOut}, kernel: {self.kernel[idx]}, stride: {self.stride[idx]}, padding: {self.padding[idx]}')
                # print(f"output height: {self.output_height}, output len: {self.output_len}")
                # if norm before conv, can use prior output shapes for norm layer
                block = nn.Sequential(nn.LayerNorm([nIn, self.output_height, self.output_len], elementwise_affine=self.ln_affine),
                                    conv2d_same.create_conv2d_pad(nIn, nOut, self.kernel[idx], stride=self.stride[idx], padding=self.padding[idx]),
                                    nn.ReLU())
            else:  
                # print(f'nIn: {nIn}, nOut: {nOut}, kernel: {self.kernel[idx]}, stride: {self.stride[idx]}, padding: {self.padding[idx]}')
                # print(f"output height: {output_height}, output len: {output_len}")
                # if norm after conv, use new output shapes for norm layer
                block = nn.Sequential(conv2d_same.create_conv2d_pad(nIn, nOut, self.kernel[idx], stride=self.stride[idx], padding=self.padding[idx]),
                                    nn.ReLU(),
                                    nn.LayerNorm([nOut, output_height, output_len], elementwise_affine=self.ln_affine))
            # update post-conv init
            self.output_height, self.output_len = output_height, output_len
            self.model_dict[f'conv_block_{idx}'] = block

            if self.pool_stride[idx] != -1:
                self.model_dict[f'hann_pool_{idx}'] = HannPooling2d(stride=self.pool_stride[idx], pool_size=self.pool_size[idx], padding=self.pool_padding[idx])
                # Compute output shapes for pooling layers using conv formula [(Height - Filter + 2Pad)/ Stride]+1
                self.output_height = int(np.floor((self.output_height - pool_size[idx][0] + 2 * pool_padding[idx][0]) / pool_stride[idx][0]) + 1)
                self.output_len = int(np.floor((self.output_len - pool_size[idx][1] + 2 * pool_padding[idx][1]) / pool_stride[idx][1]) + 1)
  
        self.output_size = self.output_height * nOut * self.output_len
        self.fullyconnected = nn.Linear(self.output_size, fc_size)
        self.relufc = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        if self.dual_task:
            self.classificationWord = nn.Linear(fc_size, self.num_words)
            self.classificationLoc = nn.Linear(fc_size, self.num_locs)
        else:
            self.classification = nn.Linear(fc_size, self.num_classes)
    

    def forward(self, cue=None, mixture=None, cue_mask_ixs=None, *args, **kwargs):
        x = torch.concat([cue, mixture], dim=1) # stack cue and mixture along channel dim
        x = self.model_dict["norm_coch_rep"](x)
        for idx in range(self.n_layers):
            x = self.model_dict[f'conv_block_{idx}'](x)
            # print(f"conv_block_{idx}, {x.max(), x.min()}")
            if self.pool_stride[idx] != -1:
                x = self.model_dict[f'hann_pool_{idx}'](x)
        x = x.view(x.size(0), self.output_size) # B x FC size
        x = self.fullyconnected(x)        
        x = self.relufc(x)
        x = self.dropout(x)        
        if self.dual_task:
            word_x = self.classificationWord(x)
            loc_x = self.classificationLoc(x)
            return word_x, loc_x
        else:
            return self.classification(x)

