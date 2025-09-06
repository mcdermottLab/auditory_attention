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
sys.path.append('phaselocknet_torch')
from phaselocknet_torch import phaselocknet_model
from phaselocknet_torch import util

class SimpleAttentionalGain(nn.Module):
    def __init__(self, frequency_dim, cnn_channels, global_avg_cue=False, n_cue_frames=None, additive=False, time_dim=-1, **kwargs):
        super(SimpleAttentionalGain, self).__init__()
        self.frequency_dim = frequency_dim
        self.time_dim = time_dim
        if global_avg_cue:
            self.ouptut_shape = (1,1)
            # self.time_average = nn.AdaptiveAvgPool2d((1, 1)) # outsize is N, C, 1, 1
        else:
            self.ouptut_shape = (frequency_dim, 1)
            # self.time_average = nn.AdaptiveAvgPool2d((frequency_dim, 1)) # outsize is N, C, FreqDim, 1
        self.bias = nn.Parameter(torch.zeros(1)) # init gain scaling to zero
        self.slope = nn.Parameter(torch.ones(1)) # init slope to one
        self.threshold = nn.Parameter(torch.zeros(1)) # init threshold to zero
        self.n_cue_frames = n_cue_frames # duration of cue in frames - full cue if None
        self.additive = additive # if True, gain is added to mixture, else multiplied
        if self.n_cue_frames:
            if n_cue_frames < 0:
                self.n_cue_frames = 1 
            print(f"Using cue duration of {self.n_cue_frames} frames")
        self.reset_parameters() 

    def reset_parameters(self):
        nn.init.constant_(self.bias, 0)
        nn.init.constant_(self.slope, 1)
        nn.init.constant_(self.threshold, 0)

    def forward(self, cue, mixture, cue_mask_ixs):
        ## Process cue 
        # time average - same as nn op, but is compat with compile 
        if self.n_cue_frames:
            n_total_frames =  cue.size(dim=self.time_dim)
            start = int((n_total_frames-self.n_cue_frames)/2)
            end = start + self.n_cue_frames
            cue = cue[...,  start : end]
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
    

class KernelAttentionalGain(nn.Module):
    def __init__(self, frequency_dim, cnn_channels, global_avg_cue=False, n_cue_frames=None, additive=False, **kwargs):
        super(KernelAttentionalGain, self).__init__()
        self.frequency_dim = frequency_dim

        if global_avg_cue:
            self.ouptut_shape = (1,1)
        else:
            self.ouptut_shape = (frequency_dim, 1)
            # self.time_average = nn.AdaptiveAvgPool2d((frequency_dim, 1)) # outsize is N, C, FreqDim, 1
        self.bias = nn.Parameter(torch.zeros(1, cnn_channels, 1, 1)) # init gain scaling to zero
        self.slope = nn.Parameter(torch.ones(1, cnn_channels, 1, 1)) # init slope to one
        self.threshold = nn.Parameter(torch.zeros(1, cnn_channels, 1, 1)) # init threshold to zero
        self.n_cue_frames = n_cue_frames # duration of cue in frames - full cue if None
        self.additive = additive # if True, gain is added to mixture, else multiplied

        if self.n_cue_frames:
            if n_cue_frames < 0:
                self.n_cue_frames = 1 
            print(f"Using cue duration of {self.n_cue_frames} frames")
        self.reset_parameters() 

    def reset_parameters(self):
        nn.init.constant_(self.bias, 0)
        nn.init.constant_(self.slope, 1)
        nn.init.constant_(self.threshold, 0)

    def forward(self, cue, mixture, cue_mask_ixs):
        ## Process cue 
        # time average - same as nn op, but is compat with compile 
        if self.n_cue_frames:
            cue = cue[...,  : self.n_cue_frames]
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
        if self.additive:
            mixture = torch.add(mixture, gain)
        else:
            mixture = torch.mul(mixture, gain)
        return mixture


class LearnedTimeAveragedGains(nn.Module):
    def __init__(self, frequency_dim, cnn_channels, time_dim, global_avg_cue=False, n_cue_frames=None, additive=False, **kwargs):
        super(LearnedTimeAveragedGains, self).__init__()
        self.frequency_dim = frequency_dim
        self.time_dim = time_dim
        # Full-set didn't work - commenting for fallback 
        # self.temporal_params = nn.Parameter(torch.ones(cnn_channels, frequency_dim, time_dim, 1)) # create time average weights
        self.temporal_params = nn.Parameter(torch.rand(time_dim, 1)) # create time average weights
        self.bias = nn.Parameter(torch.zeros(1)) # create gain scaling to zero
        self.slope = nn.Parameter(torch.ones(1)) # create slope to one
        self.threshold = nn.Parameter(torch.zeros(1)) # create threshold to zero
        self.n_cue_frames = n_cue_frames # duration of cue in frames - full cue if None
        self.additive = additive # if True, gain is added to mixture, else multiplied
        if self.n_cue_frames:
            if n_cue_frames < 0:
                self.n_cue_frames = 1 
            print(f"Using cue duration of {self.n_cue_frames} frames")
        self.reset_parameters() 

    def reset_parameters(self):
        nn.init.constant_(self.bias, 0)
        nn.init.constant_(self.slope, 1)
        nn.init.constant_(self.threshold, 0)
        # nn.init.constant_(self.temporal_params, (1/self.time_dim)) # init as standard average


    def forward(self, cue, mixture, cue_mask_ixs):
        ## Process cue 
        if self.n_cue_frames:
            cue = cue[...,  : self.n_cue_frames]
        # learned average via einsum 
        # cue = torch.einsum('bcft,cfto->bcfo', cue, self.temporal_params)  # old full-size params for ref
        cue = torch.einsum('bcft,to->bcfo', cue, self.temporal_params)
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


#### Torch module 
class ECDFGains(torch.nn.Module):
    '''
    Modified from pytorch discussion
    https://discuss.pytorch.org/t/cumulative-distribution-function-of-a-tensor-cdf/64613/2 

    Compute attentional gains using Emperical CDF of unit activity over time. 
    The emperical CDF is computed by first sorting the unit activations along 
    the time dimension, and then computing the CDF of the sorted values.
    To get a scalar gain value, the CDF is then evaluated at the unit's
    average activity. 
    '''
    def __init__(self, time_bins, side='right', time_dim=-1):
        super(ECDFGains, self).__init__()

        if side.lower() not in ['right', 'left']:
            msg = "side can take the values 'right' or 'left'"
            raise ValueError(msg)
        self.side = side
        self.time_bins = time_bins
        self.time_dim = time_dim
        # INIT cumulative probabilities
        # these will be x-axis points used to interpolate the cdf
        cdf_probs = torch.linspace(1./time_bins, 1, time_bins, dtype=torch.float)
        self.register_buffer('cdf_probs', cdf_probs)

    def forward(self, cue_tensor, mixture_tensor=None):
        # sort features of the cue tensor by time
        feat_dist = cue_tensor.sort(dim=self.time_dim).values.contiguous()
        # get average activity over time per unit. keepdim for broadcasting
        feature_means = feat_dist.mean(dim=self.time_dim, keepdim=True)
        # evaluate cdf at each mean - get ixs for interpolation
        gain_ixs = torch.searchsorted(feat_dist, feature_means, side='right') - 1 
        gains = self.cdf_probs[gain_ixs]
        if mixture_tensor is None:
            return gains
        # apply gains
        attn = mixture_tensor * gains
        return attn
    
class ECDFFeatureGains(torch.nn.Module):
    '''
    Modified from pytorch discussion
    https://discuss.pytorch.org/t/cumulative-distribution-function-of-a-tensor-cdf/64613/2 

    Compute attentional gains using Emperical CDF of unit activity over features. 
    The emperical CDF is computed by first sorting the time-averaged unit activations
    along the feature dimension, and then computing the CDF of the sorted values.
    To get a scalar gain value, the CDF is then evaluated at the unit's
    average activity. 
    '''
    def __init__(self, feature_bins, side='right', **kwargs):
        super(ECDFFeatureGains, self).__init__()

        if side.lower() not in ['right', 'left']:
            msg = "side can take the values 'right' or 'left'"
            raise ValueError(msg)
        self.side = side
        self.feature_bins = feature_bins
        # INIT cumulative probabilities
        # these will be x-axis points used to interpolate the cdf
        cdf_probs = torch.linspace(1./feature_bins, 1, feature_bins, dtype=torch.float)
        self.register_buffer('cdf_probs', cdf_probs)

    def forward(self, cue_tensor, mixture_tensor=None):
        b, c, f, t = cue_tensor.size()
        # get average activity over time per unit
        feature_means = cue_tensor.mean(dim=-1, keepdim=True).flatten(start_dim=1)
        # sort features of the cue tensor by time
        feat_dist = feature_means.sort(dim=-1).values.contiguous()
        # evaluate cdf at each mean - get ixs for interpolation
        gain_ixs = torch.searchsorted(feat_dist, feature_means, right=True) - 1 
        gains = self.cdf_probs[gain_ixs]
        # reshape 
        gains = gains.view(b, c, f, 1)
        if mixture_tensor is None:
            return gains
        # apply gains
        attn = mixture_tensor * gains
        return attn
    

class BinauralAuditoryAttentionCNNV2(nn.Module):
    def __init__(self, input_sr, out_channels, kernel, stride, padding, pool_stride, pool_size, pool_padding, attn, dropout,
                  fc_size=512, global_avg_cue=False, num_classes={"num_words":800, "num_locs":504}, frequency_dim=40,
                  residual_attn=False, n_cue_frames=None, starting_output_len = 20000, norm_first=True, ln_affine=True, 
                  v08=False, v09=False, additive=False, block_order=None, learned_gains=False, fc_attn=True, **kwargs):
        super(BinauralAuditoryAttentionCNNV2, self).__init__()
        # Setup
        print('v08', v08)
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
        self.attn = attn
        self.frequency_dim = frequency_dim
        self.n_layers = len(out_channels)
        self.norm_first = norm_first
        self.learned_gains = learned_gains
        

        # Will keep norm first to be consistent with previous models        
        if norm_first and not block_order:
            print(f"Conv block order: LN -> Conv -> ReLU")
            self.block_order = "LN -> Conv -> ReLU".lower().split(' -> ')
        elif block_order:
            print(f"Conv block order: {block_order}")
            self.block_order = block_order.lower().split(' -> ')
            if self.block_order[0] == 'ln':
                self.norm_first = True
            else:
                self.norm_first = False
            print(f"Norm first: {self.norm_first}")
        # Set to original order for backwards compat 
        elif not norm_first and not block_order:
            print(f"Conv block order: Conv -> ReLU -> LN")
            self.block_order = "Conv -> ReLU -> LN".lower().split(' -> ')
        
        if learned_gains:
            print(f"Using learned time averaged gains")

        self.input_channels = kwargs.get('input_channels', 2)
        self.n_cue_frames = n_cue_frames
        self.residual_attn = residual_attn
        self.ln_affine = ln_affine
        if residual_attn:
            print(f"Using residual attention")
        self.v08 = v08
        self.v09 = v09
        self.fc_attn = fc_attn 
        print(f"fc_attn: {fc_attn}")

        self.model_dict = nn.ModuleDict()
        self.output_height = frequency_dim
        self.output_len = starting_output_len # softcode eventually

        # build architecture
        coch_affine = self.ln_affine
        print('coch_affine:', coch_affine)
        self.model_dict["norm_coch_rep"]= nn.LayerNorm([self.input_channels, self.frequency_dim, self.output_len], elementwise_affine=coch_affine)
        # self.model_dict["attn_block_in"] = SimpleAttentionalGain(self.frequency_dim, self.input_channels, global_avg_cue=global_avg_cue, n_cue_frames=self.n_cue_frames)

        for idx in range(self.n_layers):
            # print(f"output height: {self.output_height}, output len: {self.output_len}")
            nIn = self.input_channels if idx == 0 else out_channels[idx - 1]
            nOut = out_channels[idx]
            # print(f"nIn: {nIn}, nOut: {nOut}")
            # Attentional block:
            if self.attn[idx] == 1:
                # is SimpleAttentionalGain(self.frequency_dim, self.input_channels, ... ) when ix == 0; normal for ix > 0 
                if learned_gains:
                    self.model_dict[f'attn{idx}'] = LearnedTimeAveragedGains(self.output_height, nIn, self.output_len, global_avg_cue=global_avg_cue, n_cue_frames=self.n_cue_frames, additive=additive)    
                else:
                    self.model_dict[f'attn{idx}'] = SimpleAttentionalGain(self.output_height, nIn, global_avg_cue=global_avg_cue, n_cue_frames=self.n_cue_frames, additive=additive)

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

            self.model_dict[f'conv_block_{idx}'] = self._build_cnn_block(nIn, nOut, output_height, output_len, self.kernel[idx], self.stride[idx], self.padding[idx])
            # update post-conv init
            self.output_height, self.output_len = output_height, output_len

            if self.pool_stride[idx] != -1:
                self.model_dict[f'hann_pool_{idx}'] = HannPooling2d(stride=self.pool_stride[idx], pool_size=self.pool_size[idx], padding=self.pool_padding[idx])
                # Compute output shapes for pooling layers using conv formula [(Height - Filter + 2Pad)/ Stride]+1
                self.output_height = int(np.floor((self.output_height - pool_size[idx][0] + 2 * pool_padding[idx][0]) / pool_stride[idx][0]) + 1)
                self.output_len = int(np.floor((self.output_len - pool_size[idx][1] + 2 * pool_padding[idx][1]) / pool_stride[idx][1]) + 1)
                if self.n_cue_frames:
                    self.n_cue_frames = int(np.floor((self.n_cue_frames - pool_size[idx][1] + 2 * pool_padding[idx][1]) / pool_stride[idx][1]) + 1)

        if fc_attn:
            self.model_dict[f'attnfc'] = SimpleAttentionalGain(self.output_height, nOut, global_avg_cue=global_avg_cue, n_cue_frames=self.n_cue_frames, additive=additive)

        self.output_size = self.output_height * nOut * self.output_len
        self.fullyconnected = nn.Linear(self.output_size, fc_size)
        self.relufc = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        if self.dual_task:
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

            if 'attnfc' in self.model_dict.keys():  # self.v08 or self.v09:
                attn = self.model_dict['attnfc'](cue, attn, cue_mask_ixs)
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
    
    def _build_cnn_block(self, nIn, nOut, output_height, output_len, kernel, stride, padding):
        block = nn.Sequential()
        for block_type in self.block_order:
            if block_type == 'ln':
                if self.norm_first:
                    # if norm before conv, can use prior output shapes for norm layer
                    block.add_module("ln", nn.LayerNorm([nIn, self.output_height, self.output_len], elementwise_affine=self.ln_affine))
                else:
                    # if norm after conv, use new output shapes for norm layer
                    block.add_module("ln", nn.LayerNorm([nOut, output_height, output_len], elementwise_affine=self.ln_affine))
            elif block_type == 'conv':
                block.add_module("conv", conv2d_same.create_conv2d_pad(nIn, nOut, kernel, stride=stride, padding=padding))
            elif block_type == 'relu':
                block.add_module("relu", nn.ReLU())
        return block
    


class BaselineCNNV2(nn.Module):
    def __init__(self, input_sr, out_channels, kernel, stride, padding, pool_stride, pool_size, pool_padding, attn, dropout,
                  fc_size=512, num_classes={"num_words":800, "num_locs":504}, frequency_dim=40,
                  residual_attn=False, n_cue_frames=None, starting_output_len = 20000, norm_first=True, ln_affine=True, 
                  v08=False, v09=False, additive=False, block_order=None, **kwargs):
        super(BaselineCNNV2, self).__init__()
        # Setup
        print('v08', v08)
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
        

        # Will keep norm first to be consistent with previous models        
        if norm_first and not block_order:
            print(f"Conv block order: LN -> Conv -> ReLU")
            self.block_order = "LN -> Conv -> ReLU".lower().split(' -> ')
        elif block_order:
            print(f"Conv block order: {block_order}")
            self.block_order = block_order.lower().split(' -> ')
            if self.block_order[0] == 'ln':
                self.norm_first = True
            else:
                self.norm_first = False
            print(f"Norm first: {self.norm_first}")
        # Set to original order for backwards compat 
        elif not norm_first and not block_order:
            print(f"Conv block order: Conv -> ReLU -> LN")
            self.block_order = "Conv -> ReLU -> LN".lower().split(' -> ')
 
        self.input_channels = kwargs.get('input_channels', 2)
        self.n_cue_frames = n_cue_frames
        self.residual_attn = residual_attn
        self.ln_affine = ln_affine
        if residual_attn:
            print(f"Using residual attention")

        self.model_dict = nn.ModuleDict()
        self.output_height = frequency_dim
        self.output_len = starting_output_len # softcode eventually

        # build architecture
        coch_affine = self.ln_affine
        print('coch_affine:', coch_affine)
        self.model_dict["norm_coch_rep"]= nn.LayerNorm([self.input_channels, self.frequency_dim, self.output_len], elementwise_affine=coch_affine)

        for idx in range(self.n_layers):
            # print(f"output height: {self.output_height}, output len: {self.output_len}")
            nIn = self.input_channels if idx == 0 else out_channels[idx - 1]
            nOut = out_channels[idx]
            # print(f"nIn: {nIn}, nOut: {nOut}")

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

            self.model_dict[f'conv_block_{idx}'] = self._build_cnn_block(nIn, nOut, output_height, output_len, self.kernel[idx], self.stride[idx], self.padding[idx])
            # update post-conv init
            self.output_height, self.output_len = output_height, output_len

            if self.pool_stride[idx] != -1:
                self.model_dict[f'hann_pool_{idx}'] = HannPooling2d(stride=self.pool_stride[idx], pool_size=self.pool_size[idx], padding=self.pool_padding[idx])
                # Compute output shapes for pooling layers using conv formula [(Height - Filter + 2Pad)/ Stride]+1
                self.output_height = int(np.floor((self.output_height - pool_size[idx][0] + 2 * pool_padding[idx][0]) / pool_stride[idx][0]) + 1)
                self.output_len = int(np.floor((self.output_len - pool_size[idx][1] + 2 * pool_padding[idx][1]) / pool_stride[idx][1]) + 1)
                if self.n_cue_frames:
                    self.n_cue_frames = int(np.floor((self.n_cue_frames - pool_size[idx][1] + 2 * pool_padding[idx][1]) / pool_stride[idx][1]) + 1)

        self.output_size = self.output_height * nOut * self.output_len
        self.fullyconnected = nn.Linear(self.output_size, fc_size)
        self.relufc = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        if self.dual_task:
            self.classificationWord = nn.Linear(fc_size, self.num_words)
            self.classificationLoc = nn.Linear(fc_size, self.num_locs)
        else:
            self.classification = nn.Linear(fc_size, self.num_classes)
    
    def forward(self, cue=None, mixture=None, cue_mask_ixs=None):
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
    
    def _build_cnn_block(self, nIn, nOut, output_height, output_len, kernel, stride, padding):
        block = nn.Sequential()
        for block_type in self.block_order:
            if block_type == 'ln':
                if self.norm_first:
                    # if norm before conv, can use prior output shapes for norm layer
                    block.add_module("ln", nn.LayerNorm([nIn, self.output_height, self.output_len], elementwise_affine=self.ln_affine))
                else:
                    # if norm after conv, use new output shapes for norm layer
                    block.add_module("ln", nn.LayerNorm([nOut, output_height, output_len], elementwise_affine=self.ln_affine))
            elif block_type == 'conv':
                block.add_module("conv", conv2d_same.create_conv2d_pad(nIn, nOut, kernel, stride=stride, padding=padding))
            elif block_type == 'relu':
                block.add_module("relu", nn.ReLU())
        return block

    
class BinauralAuditoryAttentionCNN(nn.Module):
    def __init__(self, input_sr, out_channels, kernel, stride, padding, pool_stride, pool_size, pool_padding, attn, dropout,
                  fc_size=512, global_avg_cue=False, num_classes={"num_words":800, "num_locs":504}, frequency_dim=40,
                  residual_attn=False, n_cue_frames=None, starting_output_len = 20000, norm_first=True, ln_affine=True, 
                  v08=False, additive=False, cue_loc_task=False, fc_attn=True,  per_kernel_gain=False, **kwargs):
        super(BinauralAuditoryAttentionCNN, self).__init__()
        # Setup
        print('v08', v08)
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
        self.attn = attn
        self.frequency_dim = frequency_dim
        self.n_layers = len(out_channels)
        self.norm_first = norm_first
        self.cue_loc_task  = cue_loc_task
        self.per_kernel_gain = per_kernel_gain
        if per_kernel_gain:
            self.gain_module = KernelAttentionalGain
            print(f"Using per-kernel gain functions")
        else:
            self.gain_module = SimpleAttentionalGain
            print(f"Using singe gain function per layer")


        if norm_first:
            print(f"Conv block order: LN -> Conv -> ReLU")
        elif not norm_first:
            print(f"Conv block order: Conv -> ReLU -> LN")

        self.input_channels = kwargs.get('input_channels', 2)
        self.n_cue_frames = n_cue_frames
        self.residual_attn = residual_attn
        self.ln_affine = ln_affine
        if residual_attn:
            print(f"Using residual attention")
        self.v08 = v08
        self.fc_attn = fc_attn
        print(f"fc_attn: {fc_attn}")

        self.model_dict = nn.ModuleDict()
        self.output_height = frequency_dim
        self.output_len = starting_output_len 

        # build architecture
        if v08:
            coch_affine = self.ln_affine
        else:
            coch_affine = True
        print('coch_affine:', coch_affine)
        self.model_dict["norm_coch_rep"]= nn.LayerNorm([self.input_channels, self.frequency_dim, self.output_len], elementwise_affine=coch_affine)
        # self.model_dict["attn_block_in"] = SimpleAttentionalGain(self.frequency_dim, self.input_channels, global_avg_cue=global_avg_cue, n_cue_frames=self.n_cue_frames)

        for idx in range(self.n_layers):

            # print(f"output height: {self.output_height}, output len: {self.output_len}")

            nIn = self.input_channels if idx == 0 else out_channels[idx - 1]
            nOut = out_channels[idx]
            # print(f"nIn: {nIn}, nOut: {nOut}")

            # Attentional block:
            if self.attn[idx] == 1:
                # is SimpleAttentionalGain(self.frequency_dim, self.input_channels, ... ) when ix == 0; normal for ix > 0 
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

class GainPostNormCNN(nn.Module):
    def __init__(self, input_sr, out_channels, kernel, stride, padding, pool_stride, pool_size, pool_padding, attn, dropout,
                  fc_size=512, global_avg_cue=False, num_classes={"num_words":800, "num_locs":504}, frequency_dim=40,
                  residual_attn=False, n_cue_frames=None, starting_output_len = 20000, norm_first=True, ln_affine=True, 
                  v08=False, additive=False, cue_loc_task=False, fc_attn=True,  per_kernel_gain=False, gain_post_norm=False, **kwargs):
        super(GainPostNormCNN, self).__init__()
        # Setup
        print('v08', v08)
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
        self.attn = attn
        self.frequency_dim = frequency_dim
        self.n_layers = len(out_channels)
        self.norm_first = norm_first
        self.cue_loc_task  = cue_loc_task
        self.per_kernel_gain = per_kernel_gain
        self.gain_post_norm = gain_post_norm

        if per_kernel_gain:
            self.gain_module = KernelAttentionalGain
            print(f"Using per-kernel gain functions")
        else:
            self.gain_module = SimpleAttentionalGain
            print(f"Using singe gain function per layer")


        if norm_first:
            print(f"Conv block order: LN -> Conv -> ReLU")
        elif not norm_first:
            print(f"Conv block order: Conv -> ReLU -> LN")

        self.input_channels = kwargs.get('input_channels', 2)
        self.n_cue_frames = n_cue_frames
        self.residual_attn = residual_attn
        self.ln_affine = ln_affine
        if residual_attn:
            print(f"Using residual attention")
        if gain_post_norm:
            print(f"Attentional gains post LayerNorm")
        self.v08 = v08
        self.fc_attn = fc_attn
        print(f"fc_attn: {fc_attn}")

        self.model_dict = nn.ModuleDict()
        self.output_height = frequency_dim
        self.output_len = starting_output_len 

        # build architecture
        if v08:
            coch_affine = self.ln_affine
        else:
            coch_affine = True
        print('coch_affine:', coch_affine)
        self.model_dict["norm_coch_rep"]= nn.LayerNorm([self.input_channels, self.frequency_dim, self.output_len], elementwise_affine=coch_affine)
        # self.model_dict["attn_block_in"] = SimpleAttentionalGain(self.frequency_dim, self.input_channels, global_avg_cue=global_avg_cue, n_cue_frames=self.n_cue_frames)

        for idx in range(self.n_layers):

            # print(f"output height: {self.output_height}, output len: {self.output_len}")

            nIn = self.input_channels if idx == 0 else out_channels[idx - 1]
            nOut = out_channels[idx]
            # print(f"nIn: {nIn}, nOut: {nOut}")
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

          
            self.model_dict[f'layernorm_{idx}'] = nn.LayerNorm([nIn, self.output_height, self.output_len], elementwise_affine=self.ln_affine)
            # Attentional block:
            if self.attn[idx] == 1:
                # is SimpleAttentionalGain(self.frequency_dim, self.input_channels, ... ) when ix == 0; normal for ix > 0 
                self.model_dict[f'attn{idx}'] = self.gain_module(self.output_height, nIn, global_avg_cue=global_avg_cue, n_cue_frames=self.n_cue_frames, additive=additive)

            self.model_dict[f'conv_block_{idx}'] = nn.Sequential(
                                                    conv2d_same.create_conv2d_pad(nIn, nOut, self.kernel[idx], stride=self.stride[idx], padding=self.padding[idx]),
                                                    nn.ReLU())
 
            # update post-conv init
            self.output_height, self.output_len = output_height, output_len

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
                mixture = self.model_dict[f'layernorm_{idx}'](mixture)
                mixture = self.model_dict[f'conv_block_{idx}'](mixture)
                # print(f"conv_block_{idx}, {mixture.mean().item(), mixture.max().item(), mixture.min().item()}")
                if self.pool_stride[idx] != -1:
                    mixture = self.model_dict[f'hann_pool_{idx}'](mixture)
            out = mixture

        else:
            cue = self.model_dict["norm_coch_rep"](cue)
            attn = self.model_dict["norm_coch_rep"](mixture)
            for idx in range(self.n_layers):
                cue = self.model_dict[f'layernorm_{idx}'](cue)
                attn = self.model_dict[f'layernorm_{idx}'](attn)
                if self.attn[idx] == 1:
                    if self.residual_attn:
                        attn = self.model_dict[f'attn{idx}'](cue, attn, cue_mask_ixs) + attn
                    else:
                        attn = self.model_dict[f'attn{idx}'](cue, attn, cue_mask_ixs)
                        
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


class BackBoneCNN(BinauralControlCNN):
    def __init__(self, input_sr, out_channels, kernel, stride, padding, pool_stride, pool_size, pool_padding, attn, dropout,
                  fc_size=512, num_classes={"num_words":800, "num_locs":504}, frequency_dim=40,
                  starting_output_len = 20000, norm_first=True, ln_affine=True, v08=False, **kwargs):
        super(BackBoneCNN, self).__init__(input_sr=input_sr,
                                          out_channels=out_channels,
                                          kernel=kernel,
                                          stride=stride,
                                          padding=padding,
                                          pool_stride=pool_stride,
                                          pool_size=pool_size,
                                          pool_padding=pool_padding,
                                          attn=attn,
                                          dropout=dropout,
                                          fc_size=fc_size,
                                          num_classes=num_classes,
                                          frequency_dim=frequency_dim,
                                          starting_output_len = starting_output_len,
                                          norm_first=norm_first,
                                          ln_affine=ln_affine,
                                          v08=v08,
                                          **kwargs) 
        
    # override forward to remove cue
    def forward(self, cue=None, mixture=None, cue_mask_ixs=None, *args, **kwargs):
        # forward will take cue as argument for compat with rest of pipeline, but won't use it. 
        x = mixture
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

class BackBoneWithECDFGains(nn.Module):
    def __init__(self, input_sr, out_channels, kernel, stride, padding, pool_stride, pool_size, pool_padding, attn, dropout,
                  fc_size=512, num_classes={"num_words":800, "num_locs":504}, frequency_dim=40,
                  starting_output_len = 20000, norm_first=True, ln_affine=True, v08=False, gain_type='ECDFGains', **kwargs):
        super(BackBoneWithECDFGains, self).__init__()
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

        # set gain function 
        if gain_type == 'ECDFGains':
            print(f"Using ECDFGains")
        elif gain_type == 'ECDFFeatureGains':
            print(f"Using ECDFFeatureGains")
        self.gain_type = gain_type
        # build architecture
        coch_affine = self.ln_affine
        self.model_dict["norm_coch_rep"]= nn.LayerNorm([self.input_channels, self.frequency_dim, self.output_len], elementwise_affine=coch_affine)

        for idx in range(self.n_layers):
            nIn = self.input_channels if idx == 0 else out_channels[idx - 1]
            nOut = out_channels[idx]
            # pre-compute conv output sizes - will assign to self.output_height and self.output_len after defining block
            # Sizes will be used for normalization layers, but depend on order of norm and conv 
            # norm -> conv gets prior output shapes, conv -> norm gets new output shapes)
            # compute output shapes using conv formula [(Height + 2Pad - dilation * (kernel - 1) -1) /  Stride] + 1
            # ignoring dilation since it's not used in this model (dilation = 1)
            if self.gain_type == 'ECDFGains':
                self.model_dict[f'attn{idx}'] = ECDFGains(self.output_len)
            elif self.gain_type == 'ECDFFeatureGains':
                self.model_dict[f'attn{idx}'] = ECDFFeatureGains(int(nIn * self.output_height))

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
  
        self.model_dict[f'attnfc'] = ECDFGains(self.output_len)
        self.output_size = self.output_height * nOut * self.output_len
        self.fullyconnected = nn.Linear(self.output_size, fc_size)
        self.relufc = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        if self.dual_task:
            self.classificationWord = nn.Linear(fc_size, self.num_words)
            self.classificationLoc = nn.Linear(fc_size, self.num_locs)
        else:
            self.classification = nn.Linear(fc_size, self.num_classes)
    

    def forward(self, cue=None, mixture=None, cue_mask_ixs=None):
        cue = self.model_dict["norm_coch_rep"](cue)
        attn = self.model_dict["norm_coch_rep"](mixture)
        for idx in range(self.n_layers):
            attn = self.model_dict[f'attn{idx}'](cue, attn)
            cue = self.model_dict[f'conv_block_{idx}'](cue)
            attn = self.model_dict[f'conv_block_{idx}'](attn)
            # print(f"conv_block_{idx} post conv and norm mean max and min: {attn.mean().item(), attn.max().item(), attn.min().item()}")
            if self.pool_stride[idx] != -1:
                cue = self.model_dict[f'hann_pool_{idx}'](cue)
                attn = self.model_dict[f'hann_pool_{idx}'](attn)

        if self.v08 and 'attnfc' in self.model_dict.keys(): 
            attn = self.model_dict['attnfc'](cue, attn)
        out = attn
        
        out = out.view(out.size(0), self.output_size) # B x FC size
        out = self.fullyconnected(out)        
        out = self.relufc(out)
        out = self.dropout(out)  
        return self.classification(out)


class BackBoneLearnedGains(nn.Module):
    def __init__(self, backbone_config, **kwargs):
        super(BackBoneLearnedGains, self).__init__()
        self.backbone_config = backbone_config
        # init and freeze backbone 
        self.backbone = BackBoneCNN(**backbone_config).eval()
        # freeze backbone 
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.model_dict = self.backbone.model_dict
        self.n_layers = len(backbone_config['out_channels'])
        for layer in range(self.n_layers):
            # gains placed before each convolutional layer
            self.model_dict[f'attn{layer}'] = SimpleAttentionalGain(None, None) 
        # add gain between last convolution and fully connected layer 
        self.model_dict['attnfc'] = SimpleAttentionalGain(None, None)

    def forward(self, cue, mixture, cue_mask_ixs=None):
        cue = self.model_dict['norm_coch_rep'](cue)
        attn = self.model_dict['norm_coch_rep'](mixture)
        for layer in range(self.n_layers):
            # apply gains
            attn = self.model_dict[f'attn{layer}'](cue, attn, cue_mask_ixs)
            # get reps from next layer
            cue = self.model_dict[f'conv_block_{layer}'](cue)
            cue = self.model_dict[f'hann_pool_{layer}'](cue)

            attn = self.model_dict[f'conv_block_{layer}'](attn)
            attn = self.model_dict[f'hann_pool_{layer}'](attn)
        # apply gains before fc layer
        out = self.model_dict['attnfc'](cue, attn, cue_mask_ixs)
        out = out.view(out.size(0), self.backbone.output_size)
        out = self.backbone.fullyconnected(out)        
        out = self.backbone.relufc(out)
        out = self.backbone.dropout(out) 
        return self.backbone.classification(out)

class SaddlerBackBoneLearnedGains(nn.Module):
    def __init__(self, dir_model, **kwargs):
        super().__init__()
        with open(os.path.join(dir_model, "config.json"), "r") as f:
            config_model = json.load(f)
        with open(os.path.join(dir_model, "arch.json"), "r") as f:
            architecture = json.load(f)
        self.backbone = phaselocknet_model.Model(
            config_model=config_model,
            architecture=architecture,
            input_shape=[2, 125_000, 2],  # <-- [batch, timesteps @ 50 kHz sampling rate, channels==2] for sound_localization
            config_random_slice={"size": [50, 20000], "buffer": [0, 500]},
        )
        util.load_model_checkpoint(
            model=self.backbone.perceptual_model,
            dir_model=dir_model,
            fn_ckpt="ckpt_BEST.pt",
            weights_only=True,
        )
        self.backbone = self.backbone.eval()
        # freeze backbone 
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.gains = nn.ModuleDict()
        self.n_layers = len([l for l in architecture if 'pool' in l['args']['name']])
        for layer in range(self.n_layers):
            # gains placed before each convolutional layer
            self.gains[f'attn{layer}'] = SimpleAttentionalGain(None, None, time_dim=-2) 

    def forward(self, cue, mixture, cue_mask_ixs=None):
        cue = self.backbone.peripheral_model(cue)
        attn = self.backbone.peripheral_model(mixture)

        for name, layer in self.backbone.perceptual_model.body.named_children():
            cue = layer(cue)
            attn = layer(attn)
            if 'pool' in name:
                block_n = re.match(r"block(\d+)_pool", name).group(1)
                attn = self.gains[f"attn{block_n}"](cue, attn, cue_mask_ixs)
        # is classification head
        logits = {name: head(attn) for name, head in self.backbone.perceptual_model.head.named_children()}
        logits = logits['label_word_int']
        return logits

class CNN2DExtractor(nn.Module):
    ''' CNN wrapper, includes relu and layer-norm if applied'''

    def __init__(self, input_sr, out_channels, kernel, stride, padding, pool_stride, pool_size, pool_padding, attn, dropout, fc_size=512,
                 global_avg_cue=False, num_classes={"num_words":998, "num_locs":504}, residual_attn=False, double_size=False, n_cue_frames=None, additive=False, **kwargs):
        super(CNN2DExtractor, self).__init__()
        # Setup
        print(f"{num_classes=}")
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
        self.frequency_dim = 40
        self.n_layers = len(out_channels)

        self.input_channels = kwargs.get('input_channels', 2)
        self.n_cue_frames = n_cue_frames
        self.residual_attn = residual_attn
        if residual_attn:
            print(f"Using residual attention")

        self.model_dict = nn.ModuleDict()
        self.output_height = self.frequency_dim
        self.output_len = 20000 # softcode eventually
        self.model_dict["norm_coch_rep"]= nn.LayerNorm([self.input_channels, self.frequency_dim, self.output_len])
        self.model_dict["attn_block_in"] = SimpleAttentionalGain(self.frequency_dim, self.input_channels, global_avg_cue=global_avg_cue, n_cue_frames=self.n_cue_frames, additive=additive)

        for idx in range(self.n_layers):
            # print(f"output height: {self.output_height}, output len: {self.output_len}")
            
            nIn = self.input_channels if idx == 0 else out_channels[idx - 1]
            nOut = out_channels[idx]
            # print(f"nIn: {nIn}, nOut: {nOut}")
            # Convolutional block:
            if self.pool_stride[idx] != -1:
                # print(f'nIn: {nIn}, nOut: {nOut}, kernel: {self.kernel[idx]}, stride: {self.stride[idx]}, padding: {self.padding[idx]}')

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
                if self.n_cue_frames:
                    self.n_cue_frames = int(np.floor((self.n_cue_frames - kernel[idx][1] + 2 * padding[idx][1]) / stride[idx][1]) + 1)
            if self.pool_stride[idx] != -1:
                # pooling layers
                self.output_height = int(np.floor((self.output_height - pool_size[idx][0] + 2 * pool_padding[idx][0]) / pool_stride[idx][0]) + 1)
                self.output_len = int(np.floor((self.output_len - pool_size[idx][1] + 2 * pool_padding[idx][1]) / pool_stride[idx][1]) + 1)
                if self.n_cue_frames:
                    self.n_cue_frames = int(np.floor((self.n_cue_frames - pool_size[idx][1] + 2 * pool_padding[idx][1]) / pool_stride[idx][1]) + 1)
            # Attentional block:
            if self.attn[idx] == 1:
                self.model_dict[f'attn{idx}'] = SimpleAttentionalGain(self.output_height, nOut, global_avg_cue=global_avg_cue, n_cue_frames=self.n_cue_frames, additive=additive)
                
        self.output_size = self.output_height * nOut * self.output_len
        self.fullyconnected = nn.Linear(self.output_size, fc_size)
        self.relufc = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        if self.dual_task:
            self.classificationWord = nn.Linear(fc_size, self.num_words)
            self.classificationLoc = nn.Linear(fc_size, self.num_locs)
        else:
            self.classification = nn.Linear(fc_size, self.num_classes)

    def forward(self, cue=None, mixture=None, cue_mask_ixs=None):
        if cue == None:
            mixture = self.model_dict["norm_coch_rep"](mixture)
            for idx in range(self.n_layers):
                mixture = self.model_dict[f'conv_block_{idx}'](mixture)
                # print(f"conv_block_{idx}, {mixture.max(), mixture.min()}")
            out = mixture

        else:
            cue = self.model_dict["norm_coch_rep"](cue)
            mixture = self.model_dict["norm_coch_rep"](mixture)
            if self.residual_attn:
                attn = self.model_dict["attn_block_in"](cue, mixture, cue_mask_ixs) + mixture
            else:
                attn = self.model_dict["attn_block_in"](cue, mixture, cue_mask_ixs)
            for idx in range(self.n_layers):
                cue = self.model_dict[f'conv_block_{idx}'](cue)
                attn = self.model_dict[f'conv_block_{idx}'](attn)
                # print('mixture acts ',  attn)
                # print(f"conv_block_{idx} pre attn, {attn.max(), attn.min()}")
                if self.attn[idx] == 1:
                    if self.residual_attn:
                        attn = self.model_dict[f'attn{idx}'](cue, attn, cue_mask_ixs) + attn
                    else:
                        attn = self.model_dict[f'attn{idx}'](cue, attn, cue_mask_ixs)
                    # print(f"conv_block_{idx} post attn, {attn.max(), attn.min()}")
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


class CueDurationCNNNew(BinauralAuditoryAttentionCNN):
    def __init__(self, input_sr, out_channels, kernel, stride, padding, pool_stride, pool_size, pool_padding, attn, dropout,
                  fc_size=512, global_avg_cue=False, num_classes={"num_words":800, "num_locs":504}, frequency_dim=40,
                  residual_attn=False, n_cue_frames=None, starting_output_len = 20000, norm_first=True, ln_affine=True, additive=False, **kwargs):
        # 1. call parent constructor
        super(CueDurationCNNNew, self).__init__(input_sr, out_channels, kernel, stride, padding,
                                             pool_stride, pool_size, pool_padding, attn, dropout,
                                             fc_size, global_avg_cue, num_classes, frequency_dim,
                                             residual_attn, n_cue_frames, starting_output_len, norm_first,
                                             ln_affine, **kwargs)
        self.model_dict = nn.ModuleDict()
        self.layer_norm_params = {}
        self.output_height = frequency_dim
        self.output_len = starting_output_len # softcode eventually
        self.n_cue_frames = n_cue_frames
        self.ln_cue_frames = n_cue_frames 
        print(n_cue_frames)
        if n_cue_frames < 5000:
            self.ln_cue_frames = 5000
        # build architecture without nn.normalization functions 

        # add init norm params 
        self.layer_norm_params[f'norm_coch_rep'] = {"weight": None, "bias": None, "cue_weight": None, "cue_bias": None, 'ln_cue_frames': self.ln_cue_frames}

        for idx in range(self.n_layers):
            nIn = self.input_channels if idx == 0 else out_channels[idx - 1]
            nOut = out_channels[idx]

            # Attentional block:
            if self.attn[idx] == 1:
                self.model_dict[f'attn{idx}'] = SimpleAttentionalGain(self.output_height, nIn, global_avg_cue=global_avg_cue, n_cue_frames=self.n_cue_frames, additive=additive)

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
                n_cue_frames = int(np.floor((self.n_cue_frames + (2 * conv_padding[1]) -  (kernel[idx][1] - 1) - 1) / stride[idx][1]) + 1)
                ln_cue_frames = int(np.floor((self.ln_cue_frames + (2 * conv_padding[1]) -  (kernel[idx][1] - 1) - 1) / stride[idx][1]) + 1)

            block = nn.Sequential(conv2d_same.create_conv2d_pad(nIn, nOut, self.kernel[idx], stride=self.stride[idx], padding=self.padding[idx]),
                                  nn.ReLU())
            
            # init layer norm params as None - will populate if in checkpoint 
            self.layer_norm_params[f'layer_norm_{idx}'] = {"weight": None, "bias": None, "cue_weight": None, "cue_bias": None, 'ln_cue_frames': self.ln_cue_frames if self.norm_first else ln_cue_frames}
            # update post-conv init
            self.output_height, self.output_len, self.n_cue_frames, self.ln_cue_frames = output_height, output_len, n_cue_frames, ln_cue_frames
            self.model_dict[f'conv_block_{idx}'] = block

            if self.pool_stride[idx] != -1:
                self.model_dict[f'hann_pool_{idx}'] = HannPooling2d(stride=self.pool_stride[idx], pool_size=self.pool_size[idx], padding=self.pool_padding[idx])
                # Compute output shapes for pooling layers using conv formula [(Height - Filter + 2Pad)/ Stride]+1
                self.output_height = int(np.floor((self.output_height - pool_size[idx][0] + 2 * pool_padding[idx][0]) / pool_stride[idx][0]) + 1)
                self.output_len = int(np.floor((self.output_len - pool_size[idx][1] + 2 * pool_padding[idx][1]) / pool_stride[idx][1]) + 1)
                self.n_cue_frames = int(np.floor((self.n_cue_frames - pool_size[idx][1] + 2 * pool_padding[idx][1]) / pool_stride[idx][1]) + 1)
                self.ln_cue_frames = int(np.floor((self.ln_cue_frames - pool_size[idx][1] + 2 * pool_padding[idx][1]) / pool_stride[idx][1]) + 1)


        self.output_size = self.output_height * nOut * self.output_len
        self.fullyconnected = nn.Linear(self.output_size, fc_size)
        self.relufc = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        # init ln params

        if self.dual_task:
            self.classificationWord = nn.Linear(fc_size, self.num_words)
            self.classificationLoc = nn.Linear(fc_size, self.num_locs)
        else:
            self.classification = nn.Linear(fc_size, self.num_classes)
    
    def forward(self, cue=None, mixture=None, cue_mask_ixs=None):
        cue = nn.functional.layer_norm(cue, cue.shape[1:],
                                       weight=self.layer_norm_params['norm_coch_rep']['cue_weight'], # [..., :  self.layer_norm_params[f'norm_coch_rep']['n_cue_frames']]
                                       bias=self.layer_norm_params['norm_coch_rep']['cue_bias'])
        
        attn = nn.functional.layer_norm(mixture, mixture.shape[1:], 
                                        weight=self.layer_norm_params['norm_coch_rep']['weight'], 
                                        bias=self.layer_norm_params['norm_coch_rep']['bias'])

        for idx in range(self.n_layers):
            if self.attn[idx] == 1:
                if self.residual_attn:
                    attn = self.model_dict[f'attn{idx}'](cue, attn, cue_mask_ixs) + attn
                else:
                    attn = self.model_dict[f'attn{idx}'](cue, attn, cue_mask_ixs)
                
            # print(f"conv_block_{idx} post gain max and min: {attn.max().item(), attn.min().item()}")
            if self.norm_first:
                cue = nn.functional.layer_norm(cue, cue.shape[1:], 
                                               weight=self.layer_norm_params[f'layer_norm_{idx}']['cue_weight'],
                                               bias=self.layer_norm_params[f'layer_norm_{idx}']['cue_bias'])
                cue = self.model_dict[f'conv_block_{idx}'](cue)
                attn = nn.functional.layer_norm(attn, attn.shape[1:], 
                                                weight=self.layer_norm_params[f'layer_norm_{idx}']['weight'],
                                                bias=self.layer_norm_params[f'layer_norm_{idx}']['bias'])
                attn = self.model_dict[f'conv_block_{idx}'](attn)
            else:
                cue = self.model_dict[f'conv_block_{idx}'](cue)
                cue = nn.functional.layer_norm(cue, cue.shape[1:], 
                                               weight=self.layer_norm_params[f'layer_norm_{idx}']['cue_weight'],
                                               bias=self.layer_norm_params[f'layer_norm_{idx}']['cue_bias'])
                attn = self.model_dict[f'conv_block_{idx}'](attn)
                attn = nn.functional.layer_norm(attn, attn.shape[1:], 
                                                weight=self.layer_norm_params[f'layer_norm_{idx}']['weight'],
                                                bias=self.layer_norm_params[f'layer_norm_{idx}']['bias'])

            # print(f"conv_block_{idx} post conv and norm max and min: {attn.max().item(), attn.min().item()}")
            if self.pool_stride[idx] != -1:
                cue = self.model_dict[f'hann_pool_{idx}'](cue)
                attn = self.model_dict[f'hann_pool_{idx}'](attn)

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
        


class CueDurationCNN2DExtractor(CNN2DExtractor):
    def __init__(self, input_sr, out_channels, kernel, stride, padding, pool_stride, pool_size, pool_padding, attn, dropout,
                  fc_size=512, global_avg_cue=False, num_classes={"num_words":998, "num_locs":504}, residual_attn=False,
                  double_size=False, n_cue_frames=None, additive=False, **kwargs):
        # 1. call parent constructor
        super(CueDurationCNN2DExtractor, self).__init__(input_sr, out_channels, kernel, stride, padding,
                                             pool_stride, pool_size, pool_padding, attn, dropout,
                                             fc_size, global_avg_cue, num_classes, residual_attn, double_size, n_cue_frames, **kwargs)
        self.model_dict = nn.ModuleDict()
        self.layer_norm_params = {}
        self.frequency_dim = 40

        self.output_height = self.frequency_dim
        self.output_len = 20000 # softcode eventually
        self.n_cue_frames = n_cue_frames
        if n_cue_frames < 5000:
            self.ln_cue_frames = 5000
        else:
            self.ln_cue_frames = self.n_cue_frames
        # build architecture without nn.normalization functions 

        # add init norm params 
        self.layer_norm_params[f'norm_coch_rep'] = {"weight": None, "bias": None, "cue_weight": None, "cue_bias": None, 'ln_cue_frames': self.ln_cue_frames}
        self.model_dict["attn_block_in"] = SimpleAttentionalGain(self.frequency_dim, self.input_channels, global_avg_cue=global_avg_cue, n_cue_frames=self.n_cue_frames, additive=additive)

        for idx in range(self.n_layers):
            nIn = self.input_channels if idx == 0 else out_channels[idx - 1]
            nOut = out_channels[idx]

            self.layer_norm_params[f'layer_norm_{idx}'] = {"weight": None, "bias": None, "cue_weight": None, "cue_bias": None, 'ln_cue_frames': self.ln_cue_frames}

            if self.pool_stride[idx] != -1:
                # print(f'nIn: {nIn}, nOut: {nOut}, kernel: {self.kernel[idx]}, stride: {self.stride[idx]}, padding: {self.padding[idx]}')

                block = nn.Sequential(
                                    conv2d_same.create_conv2d_pad(nIn, nOut, self.kernel[idx], stride=self.stride[idx], padding=self.padding[idx]),
                                    nn.ReLU(),
                                    HannPooling2d(stride=self.pool_stride[idx], pool_size=self.pool_size[idx], padding=self.pool_padding[idx]))
            else:
                block = nn.Sequential(
                                    conv2d_same.create_conv2d_pad(nIn, nOut, self.kernel[idx], stride=self.stride[idx], padding=self.padding[idx]),
                                    nn.ReLU())

            self.model_dict[f'conv_block_{idx}'] = block

            # Compute output shapes using conv formula [(Height - Filter + 2Pad)/ Stride]+1
            if self.padding[idx] == 'same':
                pass
            else:
                self.output_height = int(np.floor((self.output_height - kernel[idx][0] + 2 * padding[idx][0]) / stride[idx][0]) + 1)
                self.output_len = int(np.floor((self.output_len -  kernel[idx][1] + 2 * padding[idx][1]) / stride[idx][1]) + 1)
                if self.n_cue_frames:
                    self.n_cue_frames = int(np.floor((self.n_cue_frames - kernel[idx][1] + 2 * padding[idx][1]) / stride[idx][1]) + 1)
                    self.ln_cue_frames = int(np.floor((self.ln_cue_frames - kernel[idx][1] + 2 * padding[idx][1]) / stride[idx][1]) + 1)
            if self.pool_stride[idx] != -1:
                self.output_height = int(np.floor((self.output_height - pool_size[idx][0] + 2 * pool_padding[idx][0]) / pool_stride[idx][0]) + 1)
                self.output_len = int(np.floor((self.output_len - pool_size[idx][1] + 2 * pool_padding[idx][1]) / pool_stride[idx][1]) + 1)
                if self.n_cue_frames:
                    self.n_cue_frames = int(np.floor((self.n_cue_frames - pool_size[idx][1] + 2 * pool_padding[idx][1]) / pool_stride[idx][1]) + 1)
                    self.ln_cue_frames = int(np.floor((self.ln_cue_frames - pool_size[idx][1] + 2 * pool_padding[idx][1]) / pool_stride[idx][1]) + 1)
            # Attentional block:
            if self.attn[idx] == 1:
                self.model_dict[f'attn{idx}'] = SimpleAttentionalGain(self.output_height, nOut, global_avg_cue=global_avg_cue, n_cue_frames=self.n_cue_frames, additive=additive)
                
        self.output_size = self.output_height * nOut * self.output_len
        self.fullyconnected = nn.Linear(self.output_size, fc_size)
        self.relufc = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        if self.dual_task:
            self.classificationWord = nn.Linear(fc_size, self.num_words)
            self.classificationLoc = nn.Linear(fc_size, self.num_locs)
        else:
            self.classification = nn.Linear(fc_size, self.num_classes)
    
    def forward(self, cue=None, mixture=None, cue_mask_ixs=None):
        cue = nn.functional.layer_norm(cue, cue.shape[1:],
                                       weight=self.layer_norm_params['norm_coch_rep']['cue_weight'],
                                       bias=self.layer_norm_params['norm_coch_rep']['cue_bias'])
        
        attn = nn.functional.layer_norm(mixture, mixture.shape[1:], 
                                        weight=self.layer_norm_params['norm_coch_rep']['weight'], 
                                        bias=self.layer_norm_params['norm_coch_rep']['bias'])
        if self.residual_attn:
            attn = self.model_dict["attn_block_in"](cue, mixture, cue_mask_ixs) + mixture
        else:
            attn = self.model_dict["attn_block_in"](cue, mixture, cue_mask_ixs)

        for idx in range(self.n_layers):
            cue = nn.functional.layer_norm(cue, cue.shape[1:], 
                                            weight=self.layer_norm_params[f'layer_norm_{idx}']['cue_weight'],
                                            bias=self.layer_norm_params[f'layer_norm_{idx}']['cue_bias'])
            cue = self.model_dict[f'conv_block_{idx}'](cue)
            attn = nn.functional.layer_norm(attn, attn.shape[1:], 
                                            weight=self.layer_norm_params[f'layer_norm_{idx}']['weight'],
                                            bias=self.layer_norm_params[f'layer_norm_{idx}']['bias'])
            attn = self.model_dict[f'conv_block_{idx}'](attn)
            # print('mixture acts ',  attn)
            # print(f"conv_block_{idx} pre attn, {attn.max(), attn.min()}")
            if self.attn[idx] == 1:
                if self.residual_attn:
                    attn = self.model_dict[f'attn{idx}'](cue, attn, cue_mask_ixs) + attn
                else:
                    attn = self.model_dict[f'attn{idx}'](cue, attn, cue_mask_ixs)
                # print(f"conv_block_{idx} post attn, {attn.max(), attn.min()}")
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
        

class CueDurationCNN(BinauralAuditoryAttentionCNN):
    def __init__(self, input_sr, out_channels, kernel, stride, padding, pool_stride, pool_size, pool_padding, attn, dropout,
                  fc_size=512, global_avg_cue=False, num_classes={"num_words":800, "num_locs":504}, frequency_dim=40,
                  residual_attn=False, n_cue_frames=None, starting_output_len = 20000, norm_first=True, ln_affine=True, additive=False, **kwargs):
        # 1. call parent constructor
        super(CueDurationCNN, self).__init__(input_sr, out_channels, kernel, stride, padding,
                                             pool_stride, pool_size, pool_padding, attn, dropout,
                                             fc_size, global_avg_cue, num_classes, frequency_dim,
                                             residual_attn, n_cue_frames, starting_output_len, norm_first,
                                             ln_affine, **kwargs)
        self.model_dict = nn.ModuleDict()
        self.output_height = frequency_dim
        self.output_len = starting_output_len # softcode eventually
        self.n_cue_frames = n_cue_frames
        # build architecture without nn.normalization functions 
        for idx in range(self.n_layers):
            nIn = self.input_channels if idx == 0 else out_channels[idx - 1]
            nOut = out_channels[idx]

            # Attentional block:
            if self.attn[idx] == 1:
                self.model_dict[f'attn{idx}'] = SimpleAttentionalGain(self.output_height, nIn, global_avg_cue=global_avg_cue, n_cue_frames=self.n_cue_frames, additive=additive)

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
                    self.n_cue_frames =  int(np.floor((self.n_cue_frames + (2 * conv_padding[1]) -  (kernel[idx][1] - 1) - 1) / stride[idx][1]) + 1)

            block = nn.Sequential(conv2d_same.create_conv2d_pad(nIn, nOut, self.kernel[idx], stride=self.stride[idx], padding=self.padding[idx]),
                                  nn.ReLU())
            
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


        self.output_size = self.output_height * nOut * self.output_len
        self.fullyconnected = nn.Linear(self.output_size, fc_size)
        self.relufc = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        if self.dual_task:
            self.classificationWord = nn.Linear(fc_size, self.num_words)
            self.classificationLoc = nn.Linear(fc_size, self.num_locs)
        else:
            self.classification = nn.Linear(fc_size, self.num_classes)
    
    def forward(self, cue=None, mixture=None, cue_mask_ixs=None):
        if cue == None:
            mixture = nn.functional.layer_norm(mixture, mixture.shape[1:].prod())
            for idx in range(self.n_layers):
                if self.norm_first:
                        mixture = nn.functional.layer_norm(mixture, mixture.shape[1:].prod())
                        mixture = self.model_dict[f'conv_block_{idx}'](mixture)
                else:
                    mixture = self.model_dict[f'conv_block_{idx}'](mixture)
                    mixture = nn.functional.layer_norm(mixture, mixture.shape[1:].prod())
                # print(f"conv_block_{idx}, {mixture.max(), mixture.min()}")
                if self.pool_stride[idx] != -1:
                    mixture = self.model_dict[f'hann_pool_{idx}'](mixture)
            out = mixture

        else:
            cue = nn.functional.layer_norm(cue, cue.shape[1:])
            attn = nn.functional.layer_norm(mixture, mixture.shape[1:])

            for idx in range(self.n_layers):
                if self.attn[idx] == 1:
                    if self.residual_attn:
                        attn = self.model_dict[f'attn{idx}'](cue, attn, cue_mask_ixs) + attn
                    else:
                        attn = self.model_dict[f'attn{idx}'](cue, attn, cue_mask_ixs)
                        
                        
                # print(f"conv_block_{idx} post gain max and min: {attn.max().item(), attn.min().item()}")
                if self.norm_first:
                    cue = nn.functional.layer_norm(cue, cue.shape[1:])
                    cue = self.model_dict[f'conv_block_{idx}'](cue)
                    attn = nn.functional.layer_norm(attn, attn.shape[1:])
                    attn = self.model_dict[f'conv_block_{idx}'](attn)
                else:
                    cue = self.model_dict[f'conv_block_{idx}'](cue)
                    cue = nn.functional.layer_norm(cue, cue.shape[1:])
                    attn = self.model_dict[f'conv_block_{idx}'](attn)
                    attn = nn.functional.layer_norm(attn, attn.shape[1:])

                # print(f"conv_block_{idx} post conv and norm max and min: {attn.max().item(), attn.min().item()}")
                if self.pool_stride[idx] != -1:
                    cue = self.model_dict[f'hann_pool_{idx}'](cue)
                    attn = self.model_dict[f'hann_pool_{idx}'](attn)

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


class BaseAuditoryAttentionForTransferV1(nn.Module):
    ''' CNN wrapper, includes relu and layer-norm if applied'''

    def __init__(self, input_sr, out_channels, kernel, stride, padding, pool_stride, pool_size, pool_padding, attn, dropout,
                  global_avg_cue=False, residual_attn=False, double_size=False, n_cue_frames=None,  n_layers=None, **kwargs):
        super(BaseAuditoryAttentionForTransferV1, self).__init__()
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
        self.n_layers = n_layers

        self.input_channels = kwargs.get('input_channels', 2)
        self.n_cue_frames = n_cue_frames
        self.residual_attn = residual_attn
        if residual_attn:
            print(f"Using residual attention")

        self.model_dict = nn.ModuleDict()
        self.output_height = self.frequency_dim
        self.output_len = 20000 # softcode eventually
        self.model_dict["norm_coch_rep"]= nn.LayerNorm([self.input_channels, self.frequency_dim, self.output_len])
        self.model_dict["attn_block_in"] = SimpleAttentionalGain(self.frequency_dim, self.input_channels, global_avg_cue=global_avg_cue, n_cue_frames=self.n_cue_frames)

        for idx in range(self.n_layers):
            # print(f"output height: {self.output_height}, output len: {self.output_len}")
            
            nIn = self.input_channels if idx == 0 else out_channels[idx - 1]
            nOut = out_channels[idx]
            # print(f"nIn: {nIn}, nOut: {nOut}")
            # Convolutional block:
            if self.pool_stride[idx] != -1:
                # print(f'nIn: {nIn}, nOut: {nOut}, kernel: {self.kernel[idx]}, stride: {self.stride[idx]}, padding: {self.padding[idx]}')

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
                if self.n_cue_frames:
                    self.n_cue_frames = int(np.floor((self.n_cue_frames - kernel[idx][1] + 2 * padding[idx][1]) / stride[idx][1]) + 1)
            if self.pool_stride[idx] != -1:
                # pooling layers
                self.output_height = int(np.floor((self.output_height - pool_size[idx][0] + 2 * pool_padding[idx][0]) / pool_stride[idx][0]) + 1)
                self.output_len = int(np.floor((self.output_len - pool_size[idx][1] + 2 * pool_padding[idx][1]) / pool_stride[idx][1]) + 1)
                if self.n_cue_frames:
                    self.n_cue_frames = int(np.floor((self.n_cue_frames - pool_size[idx][1] + 2 * pool_padding[idx][1]) / pool_stride[idx][1]) + 1)
            # Attentional block:
            if self.attn[idx] == 1:
                self.model_dict[f'attn{idx}'] = SimpleAttentionalGain(self.output_height, nOut, global_avg_cue=global_avg_cue, n_cue_frames=self.n_cue_frames)
                
        self.output_size = self.output_height * nOut * self.output_len

    def forward(self, cue=None, mixture=None, cue_mask_ixs=None):
        if cue == None:
            mixture = self.model_dict["norm_coch_rep"](mixture)
            for idx in range(self.n_layers):
                mixture = self.model_dict[f'conv_block_{idx}'](mixture)
                # print(f"conv_block_{idx}, {mixture.max(), mixture.min()}")
            out = mixture

        else:
            cue = self.model_dict["norm_coch_rep"](cue)
            mixture = self.model_dict["norm_coch_rep"](mixture)
            if self.residual_attn:
                attn = self.model_dict["attn_block_in"](cue, mixture, cue_mask_ixs) + mixture
            else:
                attn = self.model_dict["attn_block_in"](cue, mixture, cue_mask_ixs)
            for idx in range(self.n_layers):
                cue = self.model_dict[f'conv_block_{idx}'](cue)
                attn = self.model_dict[f'conv_block_{idx}'](attn)
                # print('mixture acts ',  attn)
                # print(f"conv_block_{idx} pre attn, {attn.max(), attn.min()}")
                if self.attn[idx] == 1:
                    if self.residual_attn:
                        attn = self.model_dict[f'attn{idx}'](cue, attn, cue_mask_ixs) + attn
                    else:
                        attn = self.model_dict[f'attn{idx}'](cue, attn, cue_mask_ixs)
                    # print(f"conv_block_{idx} post attn, {attn.max(), attn.min()}")
            out = attn

        out = out.view(out.size(0), self.output_size) # B x FC size
        return out 


class BaseAuditoryAttentionForTransferV2(nn.Module):

    def __init__(self, input_sr, out_channels, kernel, stride, padding, pool_stride, pool_size, pool_padding, attn, dropout,
                global_avg_cue=False, frequency_dim=40, residual_attn=False, n_cue_frames=None, starting_output_len = 20000,
                norm_first=True, ln_affine=True, n_layers=None, **kwargs):
        super(BaseAuditoryAttentionForTransfer, self).__init__()
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
        if norm_first:
            print(f"Conv block order: LN -> Conv -> ReLU")
        elif not norm_first:
            print(f"Conv block order: Conv -> ReLU -> LN")

        self.input_channels = kwargs.get('input_channels', 2)
        self.n_cue_frames = n_cue_frames
        self.residual_attn = residual_attn
        self.ln_affine = ln_affine
        if residual_attn:
            print(f"Using residual attention")

        self.model_dict = nn.ModuleDict()
        self.output_height = frequency_dim
        self.output_len = starting_output_len # softcode eventually
        self.n_layers = n_layers if n_layers else len(out_channels)


        # build architecture
        self.model_dict["norm_coch_rep"]= nn.LayerNorm([self.input_channels, self.frequency_dim, self.output_len])
        # self.model_dict["attn_block_in"] = SimpleAttentionalGain(self.frequency_dim, self.input_channels, global_avg_cue=global_avg_cue, n_cue_frames=self.n_cue_frames)

        for idx in range(self.n_layers):
            nIn = self.input_channels if idx == 0 else out_channels[idx - 1]
            nOut = out_channels[idx]
            # print(f"nIn: {nIn}, nOut: {nOut}")

            # Attentional block:
            if self.attn[idx] == 1:
                # is SimpleAttentionalGain(self.frequency_dim, self.input_channels, ... ) when ix == 0; normal for ix > 0 
                self.model_dict[f'attn{idx}'] = SimpleAttentionalGain(self.output_height, nIn, global_avg_cue=global_avg_cue, n_cue_frames=self.n_cue_frames)

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
                    n_cue_frames =  int(np.floor((self.n_cue_frames + (2 * conv_padding[1]) -  (kernel[idx][1] - 1) - 1) / stride[idx][1]) + 1)

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

        self.output_size = self.output_height * nOut * self.output_len

    def forward(self, cue=None, mixture=None, cue_mask_ixs=None):
        if cue == None:
            mixture = self.model_dict["norm_coch_rep"](mixture)
            for idx in range(self.n_layers):
                mixture = self.model_dict[f'conv_block_{idx}'](mixture)
                # print(f"conv_block_{idx}, {mixture.max(), mixture.min()}")
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
                # print(f"conv_block_{idx} post gain max and min: {attn.max().item(), attn.min().item()}")
                cue = self.model_dict[f'conv_block_{idx}'](cue)
                attn = self.model_dict[f'conv_block_{idx}'](attn)
                # print(f"conv_block_{idx} post conv and norm max and min: {attn.max().item(), attn.min().item()}")
                if self.pool_stride[idx] != -1:
                    cue = self.model_dict[f'hann_pool_{idx}'](cue)
                    attn = self.model_dict[f'hann_pool_{idx}'](attn)

            out = attn
        out = out.view(out.size(0), self.output_size) # B x FC size
        return out


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
            if self.pool_stride[idx] != -1 and idx != self.n_layers - 1:
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
