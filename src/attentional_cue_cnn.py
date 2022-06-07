import torch
import numpy as np
import torch.nn as nn
from src.layers import conv2d_same, pool2d_same
from src.custom_modules import HannPooling2d


class _AttentionalCueBlock(nn.Module):
    def __init__(self, frequency_dim, cnn_channels):
        super(_AttentionalCueBlock, self).__init__()

        self.time_average = nn.AdaptiveAvgPool2d((frequency_dim, 1)) 
        self.cue_nn = nn.Linear(cnn_channels, cnn_channels) # includes bias

    def forward(self, cue, mixture):
        ## Process cue 
        cue = self.time_average(cue)
        # B x C x F x T ->  B x C x F
        cue = cue.contiguous().squeeze() # remove single time dim
        # apply cue nn and activation 
        cue = torch.sigmoid(self.cue_nn(cue)) 
        # re-add time dim for multiplication 
        cue = cue.unsqueeze(-1) 
        # Apply to mixture (element mult)
        mixture = torch.mul(mixture,cue)
        return mixture


class AttentionalCueingCNN(nn.Module):
    def __init__(self, num_classes, frequency_dim, input_channels, cnn_channels, kernel, stride, padding, pool_stride, pool_size):
        super(AttentionalCueingCNN, self).__init__()
        # Setup
        self.cnn_channels = cnn_channels
        self.stride = stride
        self.pool_size = pool_size
        self.pool_stride = pool_stride
        self.temp_downsample = self.get_downsample_factor()
    
        self.cnn = nn.Sequential()
        self.output_height = frequency_dim # initialization for output feature dim calculation
        self.output_len = int(8000 * 2) # init samples fixed - 2 seconds at 8kHz - softcode eventually
    
        n_layers = len(cnn_channels)

        # Add convolutional layers 
        for l in range(n_layers):
            nIn = input_channels if l == 0 else cnn_channels[l - 1]
            nOut = cnn_channels[l]
            # LayerNorm
            self.cnn.add_module('layernorm{0}'.format(l), nn.LayerNorm([nIn, self.output_height, self.output_len]))  
            # Convolution
            self.cnn.add_module('conv{0}'.format(l),
                           conv2d_same.create_conv2d_pad(nIn, nOut, kernel_size=kernel[l], stride=stride[l], padding=padding[l]))
            # Activation 
            self.cnn.add_module('relu{0}'.format(l), nn.ReLU(inplace = True))
            # Pooling 
            if l < n_layers: # use average pooling at end of CNN instead 
                pool_padding = [pad//2  if pad > 1 else 0 for pad in pool_size[l]]
                self.cnn.add_module('pooling{0}'.format(l),
                        HannPooling2d(stride=pool_stride[l], pool_size=pool_size[l], padding=pool_padding))

            # Compute output shapes using conv formula [(Height - Filter + 2Pad)/ Stride]+1
            # conv layers:
            if padding[l] != 'same':
                if padding[l] == 'valid':
                    padding[l] = 0 
                self.output_height = int((np.floor(self.output_height - kernel[l][0] + 2 * padding[l]) / stride[l]) + 1)
                self.output_len = int((np.floor(self.output_len -  kernel[l][1] + 2 * padding[l]) / stride[l]) + 1)
#             # pooling layers
            self.output_height = int((np.floor(self.output_height - pool_size[l][0] + 2 * pool_padding[0]) / pool_stride[l][0]) + 1)
            self.output_len = int((np.floor(self.output_len - pool_size[l][1] + 2 * pool_padding[1]) / pool_stride[l][1]) + 1)

        self.avgpool =  pool2d_same.create_pool2d('avg', kernel_size = [2,5] , stride = [2,2], padding = 'same')
        
        self.output_height = int((np.floor(self.output_height - 2) / 2) + 1)
        self.output_len = int((np.floor(self.output_len - 5 + 2*2) / 2) + 1) # avepool adds padding of 2
        
        self.output_size = self.output_height * nOut * self.output_len
        
        self.fc =  nn.Linear(self.output_size, 4094)
        self.relufc = nn.ReLU(inplace = True)
        self.dropout = nn.Dropout()
        self.logits = nn.Linear(4094, num_classes)

    def get_downsample_factor(self):
        # Get total temporal downsampling factor. Convolutional kernel stride is an int.
        # pool stride is a 2 element list where the 2nd element is the stride in time
        return int(np.prod([k_s * p_s[1] for k_s, p_s in zip(self.stride, self.pool_stride)]))

    def forward(self, cue, mixture=None):
        for module in self.cnn:

        batch_size = feature.size(0)
        # forward through cnn layers
        feature = self.cnn(feature)
        feature = self.avgpool(feature)
        # B x C x H X W -> B x C*H*W
        feature = feature.view(batch_size, -1) 
        feature = self.fc(feature)
        feature = self.relufc(feature)
        feature = self.dropout(feature)
        feature = self.logits(feature) # now logits

        return feature