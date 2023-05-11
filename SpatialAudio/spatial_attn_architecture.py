import torch
import numpy as np
import torch.nn as nn
from src.layers import conv2d_same
from src.custom_modules import HannPooling2d

class SimpleAttentionalGain(nn.Module):
    def __init__(self, frequency_dim, cnn_channels):
        super(SimpleAttentionalGain, self).__init__()
        self.time_average = nn.AdaptiveAvgPool2d((frequency_dim, 1)) # outsize is N, C, FreqDim, 1
        self.bias = nn.Parameter(torch.zeros(1)) # init gain scaling to zero
        self.slope = nn.Parameter(torch.ones(1)) # init slope to one
        self.threshold = nn.Parameter(torch.zeros(1)) # init threshold to zero
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

class CNN2DExtractor(nn.Module):
    ''' CNN wrapper, includes relu and layer-norm if applied'''

    def __init__(self, input_sr, input_channels, out_channels, kernel, stride, padding, pool_stride, pool_size):
        super(CNN2DExtractor, self).__init__()
        # Setup
        self.out_channels = out_channels
        self.stride = stride
        self.pool_size = pool_size
        self.pool_stride = pool_stride
        self.frequency_dim = 50
        self.input_sr = input_sr

        n_layers = len(out_channels)

        self.cnn = nn.Sequential()
        self.output_height = self.frequency_dim # initialization for output feature dim calculation
        self.output_len = int(self.input_sr * 10) # init samples are 10 seconds at 8kHz - softcode eventually

        for l in range(n_layers):
            nIn = 1 if l == 0 else out_channels[l - 1]
            nOut = out_channels[l]
            # LayerNorm
            self.cnn.add_module('layernorm{0}'.format(l), nn.LayerNorm([nIn, self.output_height, self.output_len]))  
            # Convolution
            self.cnn.add_module('conv{0}'.format(l),
                           conv2d_same.create_conv2d_pad(nIn, nOut, kernel[l], stride=stride[l], padding=padding[l]))
            # Activation 
            self.cnn.add_module('relu{0}'.format(l), nn.ReLU())
            # Pooling 
            self.cnn.add_module('pooling{0}'.format(l),
                      HannPooling2d(stride=pool_stride[l], pool_size=pool_size[l]))

            # Compute output shapes using conv formula [(Height - Filter + 2Pad)/ Stride]+1
            if padding[l] == 'same':
                self.output_height = int((self.output_height / stride[l]) + 1)
                self.output_len = int((self.output_len / stride[l]) + 1)
            else:
                self.output_height = int(np.floor((self.output_height - kernel[l][0] + 2 * padding[l]) / stride[l]) + 1)
                self.output_len = int(np.floor((self.output_len -  kernel[l][1] + 2 * padding[l]) / stride[l]) + 1)
            # pooling layers
            self.output_height = int(np.floor((self.output_height - pool_size[l][0]) / pool_stride[l][0]) + 1)
            self.output_len = int(np.floor((self.output_len - pool_size[l][1]) / pool_stride[l][1]) + 1)
        self.output_size = self.output_height * nOut
        self.temp_downsample = self.get_downsample_factor()

    def forward(self, cue, mixture): 
         # pass cue through cnn & store reps
        if cue == None:
            mixture = self.norm_coch_rep(mixture)
            mixture = self.conv0(mixture) # has layer norm as 1st layer - may be a problem? 
            mixture = self.conv1(mixture)
            mixture = self.conv2(mixture)
            mixture = self.conv3(mixture)
            mixture = self.conv4(mixture)
            mixture = self.conv5(mixture)
            out = self.conv6(mixture)

        else:
            cue = self.norm_coch_rep(cue)
            cue0 = self.conv0(cue) # has layer norm as 1st layer - may be a problem? 
            cue1 = self.conv1(cue0)
            cue2 = self.conv2(cue1)
            cue3 = self.conv3(cue2)
            cue4 = self.conv4(cue3)
            cue5 = self.conv5(cue4)
            cue6 = self.conv6(cue5)

            ## Combine cue and mixture using attention
            mixture = self.norm_coch_rep(mixture)
            # attn for cochlear model
            attn = self.attn_block_in(cue, mixture)
            # conv 0 
            attn = self.conv0(attn)
            attn = self.attn_block0(cue0, attn)
            # conv 1
            attn = self.conv1(attn)
            attn = self.attn_block1(cue1, attn)
            #conv 2
            attn = self.conv2(attn)
            attn = self.attn_block2(cue2, attn)
            #conv 3
            attn = self.conv3(attn)
            attn = self.attn_block3(cue3, attn)
            #conv4
            attn = self.conv4(attn)
            attn = self.attn_block4(cue4, attn)
            #conv5
            attn = self.conv5(attn)
            attn = self.attn_block5(cue5, attn)
            #conv6
            attn = self.conv6(attn)
            attn = self.attn_block6(cue6, attn)

            out = attn

        out = out.view(out.size(0), 512*6*16) # B x FC size
        out = self.fullyconnected(out)        
        out = self.relufc(out)
        out = self.dropout(out)        
        out = self.classification(out)
        return out
