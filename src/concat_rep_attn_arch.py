import torch
import numpy as np
import torch.nn as nn
from src.layers import conv2d_same
from src.custom_modules import HannPooling2d


class AuditoryCNN(nn.Module):
    def __init__(self, num_classes=1000, fc_size=4096, input_width = 16000, global_avg=False):
        super(AuditoryCNN, self).__init__()
        layer_width = input_width

        self.conv0 = nn.Sequential(
                    nn.LayerNorm([1, 40, layer_width]),
                    conv2d_same.create_conv2d_pad(1, 32, kernel_size = [2, 34], stride = [1, 1], padding = 'same'),
                    nn.ReLU(inplace = True),
                    HannPooling2d(stride = [2, 4], pool_size = [9, 13], padding = [4, 6])
        )

        layer_width = np.ceil(layer_width / 4).astype('int') # 4 is tmporal downsampling factor in pool (eg, stride[1] in last pool layer)

        self.conv1 = nn.Sequential(
            nn.LayerNorm([32, 20, layer_width]),
            conv2d_same.create_conv2d_pad(32, 64, kernel_size = [2, 14], stride = [1,1], padding = 'same'),
            nn.ReLU(inplace = True),
            HannPooling2d(stride = [2, 4], pool_size = [9, 13], padding = [4, 6])
        )

        layer_width = np.ceil(layer_width / 4).astype('int') # 4 is tmporal downsampling factor in pool (eg, stride[1] in last pool layer)

        self.conv2 = nn.Sequential(
            nn.LayerNorm([64, 10, layer_width]),
            conv2d_same.create_conv2d_pad(64, 256, kernel_size = [5, 5], stride = [1,1], padding = 'same'),
            nn.ReLU(inplace = True),
            HannPooling2d(stride = [1, 4 if layer_width % 4 == 0 else 5], pool_size = [1, 13], padding = [0, 6])
        )

        self.conv3 =  nn.Sequential(
            nn.LayerNorm([256, 10, 250]),
            conv2d_same.create_conv2d_pad(256, 512, kernel_size = [5, 5], stride = [1,1], padding = 'same'),
            nn.ReLU(inplace = True),
            HannPooling2d(stride = [1, 4], pool_size = [1, 13], padding = [0, 6])
        )

        self.conv4 = nn.Sequential(
            nn.LayerNorm([512, 10, 63]),
            conv2d_same.create_conv2d_pad(512, 512, kernel_size = [6, 6], stride = [1,1], padding = 'same'),
            nn.ReLU(inplace = True),
            HannPooling2d(stride = [1, 1], pool_size = [1, 1], padding = [0, 0])
        )

        self.conv5 = nn.Sequential(
            nn.LayerNorm([512, 10, 63]),
            conv2d_same.create_conv2d_pad(512, 512, kernel_size = [5, 5], stride = [1,1], padding = 'same'),
            nn.ReLU(inplace = True),
            HannPooling2d(stride = [1, 1], pool_size = [1, 1], padding = [0, 0])
        )

        self.conv6 = nn.Sequential(
            nn.LayerNorm([512, 10, 63]),
            conv2d_same.create_conv2d_pad(512, 512, kernel_size = [6, 6], stride = [1,1], padding = 'same'),
            nn.ReLU(inplace = True),
            HannPooling2d(stride = [2, 4], pool_size = [6, 13], padding = [3, 6])
        )

        self.fullyconnected = nn.Linear(512*6*16*2, fc_size) # is last conv shape * 2 x fc_size
        self.relufc = nn.ReLU()
        self.dropout = nn.Dropout()
        self.classification = nn.Linear(fc_size, num_classes)
        

    def forward(self, cue, mixture=None):
        # pass cuethrough cnn & store reps
        cue = self.conv0(cue) # has layer norm as 1st layer - may be a problem? 
        cue = self.conv1(cue)
        cue = self.conv2(cue)
        cue = self.conv3(cue)
        cue = self.conv4(cue)
        cue = self.conv5(cue)
        cue = self.conv6(cue)
        
        ## Combine cueand mixture using attention
        mixture = self.conv0(mixture)
        mixture = self.conv1(mixture)
        mixture = self.conv2(mixture)
        mixture = self.conv3(mixture)
        mixture = self.conv4(mixture)
        mixture = self.conv5(mixture)
        mixture = self.conv6(mixture)

        # concat cue and mixture
        mixture = mixture.view(mixture.size(0), 512*6*16) # B x FC size
        cue = cue.view(cue.size(0), 512*6*16) # B x FC size
        out = torch.cat([mixture,cue], dim=1)
        out = self.fullyconnected(out)        
        out = self.relufc(out)
        out = self.dropout(out)        
        out = self.classification(out)
        return out
        
    