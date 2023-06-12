import torch
import numpy as np
import torch.nn as nn
from src.layers import conv2d_same
from src.custom_modules import HannPooling2d

class SimpleAttentionalGain(nn.Module):
    def __init__(self, frequency_dim, cnn_channels, global_avg=False):
        super(SimpleAttentionalGain, self).__init__()
        if global_avg:
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


class AuditoryCNN(nn.Module):
    def __init__(self, num_classes=1000, fc_size=4096, global_avg=False, **kwargs):
        super(AuditoryCNN, self).__init__()

        self.norm_coch_rep = nn.LayerNorm([1, 40, 16000])
        # self.attn_block_in = SimpleAttentionalGain(40, 1)

        self.conv0 = nn.Sequential(
                    nn.LayerNorm([1, 40, 16000]),
                    conv2d_same.create_conv2d_pad(1, 32, kernel_size = [2, 34], stride = [1, 1], padding = 'same'),
                    nn.ReLU(inplace = True),
                    HannPooling2d(stride = [2, 4], pool_size = [9, 13], padding = [4, 6])
        )
        # self.attn_block0 = SimpleAttentionalGain(20, 32)

        self.conv1 = nn.Sequential(
            nn.LayerNorm([32, 20, 4000]),
            conv2d_same.create_conv2d_pad(32, 64, kernel_size = [2, 14], stride = [1,1], padding = 'same'),
            nn.ReLU(inplace = True),
            HannPooling2d(stride = [2, 4], pool_size = [9, 13], padding = [4, 6])
        )
        # self.attn_block1 = SimpleAttentionalGain(10, 64)

        self.conv2 = nn.Sequential(
            nn.LayerNorm([64, 10, 1000]),
            conv2d_same.create_conv2d_pad(64, 256, kernel_size = [5, 5], stride = [1,1], padding = 'same'),
            nn.ReLU(inplace = True),
            HannPooling2d(stride = [1, 4], pool_size = [1, 13], padding = [0, 6])
        )
        # self.attn_block2 = SimpleAttentionalGain(10, 256)

        self.conv3 =  nn.Sequential(
            nn.LayerNorm([256, 10, 250]),
            conv2d_same.create_conv2d_pad(256, 512, kernel_size = [5, 5], stride = [1,1], padding = 'same'),
            nn.ReLU(inplace = True),
            HannPooling2d(stride = [1, 4], pool_size = [1, 13], padding = [0, 6])
        )
        # self.attn_block3 = SimpleAttentionalGain(10, 512)

        self.conv4 = nn.Sequential(
            nn.LayerNorm([512, 10, 63]),
            conv2d_same.create_conv2d_pad(512, 512, kernel_size = [6, 6], stride = [1,1], padding = 'same'),
            nn.ReLU(inplace = True),
            HannPooling2d(stride = [1, 1], pool_size = [1, 1], padding = [0, 0])
        )
        # self.attn_block4 = SimpleAttentionalGain(10, 512)

        self.conv5 = nn.Sequential(
            nn.LayerNorm([512, 10, 63]),
            conv2d_same.create_conv2d_pad(512, 512, kernel_size = [5, 5], stride = [1,1], padding = 'same'),
            nn.ReLU(inplace = True),
            HannPooling2d(stride = [1, 1], pool_size = [1, 1], padding = [0, 0])
        )
        # self.attn_block5 = SimpleAttentionalGain(10, 512)

        self.conv6 = nn.Sequential(
            nn.LayerNorm([512, 10, 63]),
            conv2d_same.create_conv2d_pad(512, 512, kernel_size = [6, 6], stride = [1,1], padding = 'same'),
            nn.ReLU(inplace = True),
            HannPooling2d(stride = [2, 4], pool_size = [6, 13], padding = [3, 6])
        )
        self.attn_block6 = SimpleAttentionalGain(6, 512, global_avg=global_avg)

        self.fullyconnected = nn.Linear(512*6*16, fc_size)
        self.relufc = nn.ReLU(inplace = True)
        self.dropout = nn.Dropout()
        self.classification = nn.Linear(fc_size, num_classes)
        

    def forward(self, cue, mixture=None):
        # pass cue through cnn & store reps
        cue = self.norm_coch_rep(cue)
        cue0 = self.conv0(cue) # has layer norm as 1st layer - may be a problem? 
        cue1 = self.conv1(cue0)
        cue2 = self.conv2(cue1)
        cue3 = self.conv3(cue2)
        cue4 = self.conv4(cue3)
        cue5 = self.conv5(cue4)
        cue6 = self.conv6(cue5)
        
        ## Combine cue and mixture using attention
        if mixture is not None:
            mixture = self.norm_coch_rep(mixture)
            # conv 0 
            mixture = self.conv0(mixture)
            # conv 1
            mixture = self.conv1(mixture)
            #conv 2
            mixture = self.conv2(mixture)
            #conv 3
            mixture = self.conv3(mixture)
            # mixture = self.mixture_block3(cue3, mixture)
            #conv4
            mixture = self.conv4(mixture)
            # mixture = self.mixture_block4(cue4, mixture)
            #conv5
            mixture = self.conv5(mixture)
            # mixture = self.mixture_block5(cue5, mixture)
            #conv6
            mixture = self.conv6(mixture)
            mixture = self.attn_block6(cue6, mixture)

            out = mixture
        else:
            out = cue6

        out = out.view(out.size(0), 512*6*16) # B x FC size
        out = self.fullyconnected(out)        
        out = self.relufc(out)
        out = self.dropout(out)        
        out = self.classification(out)
        return out
        
    