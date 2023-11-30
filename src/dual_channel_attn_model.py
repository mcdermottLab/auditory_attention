import torch
import numpy as np
import torch.nn as nn
from src.layers import conv2d_same
from src.custom_modules import HannPooling2d


class AuditoryCNN(nn.Module):
    def __init__(self, num_classes=1000, input_width = 16000, fc_size=4096, **kwargs):
        super(AuditoryCNN, self).__init__()

        layer_width = input_width

        self.conv0 = nn.Sequential(
                    nn.LayerNorm([2, 40, layer_width]),
                    conv2d_same.create_conv2d_pad(2, 32, kernel_size = [2, 34], stride = [1, 1], padding = 'same'),
                    nn.ReLU(),
                    HannPooling2d(stride = [2, 4], pool_size = [9, 13], padding = [4, 6])
        )

        layer_width = np.ceil(layer_width / 4).astype('int') # 4 is tmporal downsampling factor in pool (eg, stride[1] in last pool layer)

        self.conv1 = nn.Sequential(
            nn.LayerNorm([32, 20, layer_width]),
            conv2d_same.create_conv2d_pad(32, 64, kernel_size = [2, 14], stride = [1,1], padding = 'same'),
            nn.ReLU(),
            HannPooling2d(stride = [2, 4], pool_size = [9, 13], padding = [4, 6])
        )

        layer_width = np.ceil(layer_width / 4).astype('int') # 4 is tmporal downsampling factor in pool (eg, stride[1] in last pool layer)

        self.conv2 = nn.Sequential(
            nn.LayerNorm([64, 10, layer_width]),
            conv2d_same.create_conv2d_pad(64, 256, kernel_size = [5, 5], stride = [1,1], padding = 'same'),
            nn.ReLU(),
            HannPooling2d(stride = [1, 4 if layer_width % 4 == 0 else 5], pool_size = [1, 13], padding = [0, 6])
        )

        self.conv3 =  nn.Sequential(
            nn.LayerNorm([256, 10, 250]),
            conv2d_same.create_conv2d_pad(256, 512, kernel_size = [5, 5], stride = [1,1], padding = 'same'),
            nn.ReLU(),
            HannPooling2d(stride = [1, 4], pool_size = [1, 13], padding = [0, 6])
        )

        self.conv4 = nn.Sequential(
            nn.LayerNorm([512, 10, 63]),
            conv2d_same.create_conv2d_pad(512, 512, kernel_size = [6, 6], stride = [1,1], padding = 'same'),
            nn.ReLU(),
            HannPooling2d(stride = [1, 1], pool_size = [1, 1], padding = [0, 0])
        )

        self.conv5 = nn.Sequential(
            nn.LayerNorm([512, 10, 63]),
            conv2d_same.create_conv2d_pad(512, 512, kernel_size = [5, 5], stride = [1,1], padding = 'same'),
            nn.ReLU(),
            HannPooling2d(stride = [1, 1], pool_size = [1, 1], padding = [0, 0])
        )

        self.conv6 = nn.Sequential(
            nn.LayerNorm([512, 10, 63]),
            conv2d_same.create_conv2d_pad(512, 512, kernel_size = [6, 6], stride = [1,1], padding = 'same'),
            nn.ReLU(),
            HannPooling2d(stride = [2, 4], pool_size = [6, 13], padding = [3, 6])
        )

        self.fullyconnected = nn.Linear(512*6*16, fc_size) # is last conv shape * 2 x fc_size
        self.relufc = nn.ReLU()
        self.dropout = nn.Dropout()
        self.classification = nn.Linear(fc_size, num_classes)
        

    def forward(self, cue, mixture, *args):
        x = torch.concat([cue, mixture], dim=1) # stack cue and mixture along channel dim
        x = self.conv0(x) # has layer norm as 1st layer - may be a problem? 
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)

        x = x.view(x.size(0), x.shape[1:].numel()) # B x FC size
        x = self.fullyconnected(x)        
        x = self.relufc(x)
        x = self.dropout(x)        
        x = self.classification(x)
        return x
        
    