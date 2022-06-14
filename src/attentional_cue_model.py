import torch
import numpy as np
import torch.nn as nn
from src.layers import conv2d_same
from src.custom_modules import HannPooling2d



class _AttentionalCueBlock(nn.Module):
    def __init__(self, frequency_dim, cnn_channels):
        super(_AttentionalCueBlock, self).__init__()

        self.time_average = nn.AdaptiveAvgPool2d((frequency_dim, 1)) 
        self.cue_nn = nn.Linear(cnn_channels, cnn_channels) # includes bias

    def forward(self, cue, mixture):
        ## Process cue 
        cue = self.time_average(cue)
        # B x C x F x T ->  B x C x F x 1
        cue = cue.contiguous().squeeze() # remove single time dim
        # apply cue nn and activation 
        cue = torch.sigmoid(self.cue_nn(cue)) 
        # re-add time dim for multiplication 
        cue = cue.unsqueeze(-1) 
        # Apply to mixture (element mult)
        mixture = torch.mul(mixture,cue)
        return mixture

class _SimpleAttentionalCueBlock(nn.Module):
    def __init__(self, frequency_dim, cnn_channels):
        super(_SimpleAttentionalCueBlock, self).__init__()
        self.time_average = nn.AdaptiveAvgPool2d((frequency_dim, 1)) 

    def forward(self, cue, mixture):
        ## Process cue 
        cue = self.time_average(cue)
        # activate 
        cue = torch.sigmoid(cue) # may want to try softmax 
        # Apply to mixture (element mult)
        mixture = torch.mul(mixture, cue)
        return mixture

class AuditoryCNN(nn.Module):
    def __init__(self, num_classes=1000):
        super(AuditoryCNN, self).__init__()

        self.attn_block_in = _SimpleAttentionalCueBlock(40, 1)

        self.conv0 = nn.Sequential(
                    nn.LayerNorm([1, 40, 16000]),
                    conv2d_same.create_conv2d_pad(1, 32, kernel_size = [2, 34], stride = [1, 1], padding = 'same'),
                    nn.ReLU(inplace = True),
                    HannPooling2d(stride = [2, 4], pool_size = [9, 13], padding = [4, 6])
        )
        self.attn_block0 = _SimpleAttentionalCueBlock(20, 32)

        self.conv1 = nn.Sequential(
            nn.LayerNorm([32, 20, 4000]),
            conv2d_same.create_conv2d_pad(32, 64, kernel_size = [2, 14], stride = [1,1], padding = 'same'),
            nn.ReLU(inplace = True),
            HannPooling2d(stride = [2, 4], pool_size = [9, 13], padding = [4, 6])
        )
        self.attn_block2 = _SimpleAttentionalCueBlock(10, 64)

        self.conv2 = nn.Sequential(
            nn.LayerNorm([64, 10, 1000]),
            conv2d_same.create_conv2d_pad(64, 256, kernel_size = [5, 5], stride = [1,1], padding = 'same'),
            nn.ReLU(inplace = True),
            HannPooling2d(stride = [1, 4], pool_size = [1, 13], padding = [0, 6])
        )
        self.attn_block2 = _SimpleAttentionalCueBlock(10, 256)

        self.con3 =  nn.Sequential(
            nn.LayerNorm([256, 10, 250]),
            conv2d_same.create_conv2d_pad(256, 512, kernel_size = [5, 5], stride = [1,1], padding = 'same'),
            nn.ReLU(inplace = True),
            HannPooling2d(stride = [1, 4], pool_size = [1, 13], padding = [0, 6])
        )
        self.attn_block3 = _SimpleAttentionalCueBlock(10, 512)

        self.conv4 = nn.Sequential(
            nn.LayerNorm([512, 10, 63]),
            conv2d_same.create_conv2d_pad(512, 512, kernel_size = [6, 6], stride = [1,1], padding = 'same'),
            nn.ReLU(inplace = True),
            HannPooling2d(stride = [1, 1], pool_size = [1, 1], padding = [0, 0])
        )
        self.attn_block4 = _SimpleAttentionalCueBlock(10, 512)

        self.conv5 = nn.Sequential(
            nn.LayerNorm([512, 10, 63]),
            conv2d_same.create_conv2d_pad(512, 512, kernel_size = [5, 5], stride = [1,1], padding = 'same'),
            nn.ReLU(inplace = True),
            HannPooling2d(stride = [1, 1], pool_size = [1, 1], padding = [0, 0])
        )
        self.attn_block5 = _SimpleAttentionalCueBlock(10, 512)

        self.conv6 = nn.Sequential(
            nn.LayerNorm([512, 10, 63]),
            conv2d_same.create_conv2d_pad(512, 512, kernel_size = [6, 6], stride = [1,1], padding = 'same'),
            nn.ReLU(inplace = True),
            HannPooling2d(stride = [2, 4], pool_size = [6, 13], padding = [3, 6])
        )
        self.attn_block6 = _SimpleAttentionalCueBlock(5, 512)

        self.fullyconnected = nn.Linear(512*5*16, 4096)
        self.relufc = nn.ReLU(inplace = True)
        self.dropout = nn.Dropout()
        self.classification = nn.Linear(4096, num_classes)
        

    def forward(self, cue, mixture=None):
        # pass cue through cnn
        cue0 = self.conv0(cue)
        cue1 = self.conv1(cue0)
        cue2 = self.conv2(cue1)
        cue3 = self.conv3(cue2)
        cue4 = self.conv4(cue3)
        cue5 = self.conv5(cue4)
        cue6 = self.conv6(cue5)
        
        ## Combine cue and mixture using attention
        if mixture:
            # attn from coch
            mixture = self.attn_block_in(cue, mixture)
            # conv 0 
            mix0 = self.conv0(mixture)
            attn0 = self.attn_block0(cue0, mix0)
            # conv 1
            mix1 = self.conv1(mix0)
            attn1 = self.conv1(attn0)
            # add attention as residual connection  
            attn1 += self.attn_block1(cue1, mix1)
            #conv 2
            mix2 = self.conv2(mix1)
            attn2 = self.conv2(attn1)
            attn2 += self.attn_block2(cue2, mix2)
            #conv 3
            mix3 = self.conv3(mix2)
            attn3 = self.conv3(attn2)
            attn3 += self.attn_block3(cue3, mix3)
            #conv4
            mix4 = self.conv4(mix3)
            attn4 = self.conv4(attn3)
            attn4 += self.attn_block4(cue4, mix4)
            #conv5
            mix5 = self.conv5(mix4)
            attn5 = self.conv5(attn4)
            attn5 += self.attn_block5(cue5, mix5)
            #conv6
            mix6 = self.conv6(mix5)
            attn6 = self.conv6(attn5)
            attn6 += self.attn_block6(cue6, mix6)

            out = attn6
        else:
            out = cue6
        
        out = out.view(out.size(0), 512*5*16) # B x FC size
        out = self.fullyconnected(out)        
        out = self.relufc(out)
        out = self.dropout(out)        
        out = self.classification(out)
        return out
        
    