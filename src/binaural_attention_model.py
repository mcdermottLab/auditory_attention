import torch
import numpy as np
import torch.nn as nn
from src.layers import conv2d_same
from src.custom_modules import HannPooling2d


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

    def forward(self, cue, mixture, cue_mask_ixs):
        ## Process cue 
        cue = self.time_average(cue)
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


class BinauralAttentionCNN(nn.Module):
    def __init__(self, num_classes={'num_words':998, 'num_locs':504}, fc_size=512, global_avg_cue=False, **kwargs):
        super(BinauralAttentionCNN, self).__init__()

        # setup vars for task
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

        self.norm_coch_rep = nn.LayerNorm([2, 40, 20000])
        self.attn_block_in = SimpleAttentionalGain(40, 2, global_avg_cue=global_avg_cue)

        self.conv0 = nn.Sequential(
                    nn.LayerNorm([2, 40, 20000]),
                    conv2d_same.create_conv2d_pad(2, 32, kernel_size = [2, 34], stride = [1, 1], padding = 'same'),
                    nn.ReLU(),
                    HannPooling2d(stride = [2, 4], pool_size = [9, 13], padding = [4, 6])
        )
        self.attn_block0 = SimpleAttentionalGain(20, 32, global_avg_cue=global_avg_cue)

        self.conv1 = nn.Sequential(
            nn.LayerNorm([32, 20, 5000]),
            conv2d_same.create_conv2d_pad(32, 64, kernel_size = [2, 14], stride = [1,1], padding = 'same'),
            nn.ReLU(),
            HannPooling2d(stride = [2, 4], pool_size = [9, 13], padding = [4, 6])
        )
        self.attn_block1 = SimpleAttentionalGain(10, 64, global_avg_cue=global_avg_cue)

        self.conv2 = nn.Sequential(
            nn.LayerNorm([64, 10, 1250]),
            conv2d_same.create_conv2d_pad(64, 256, kernel_size = [5, 5], stride = [1,1], padding = 'same'),
            nn.ReLU(),
            HannPooling2d(stride = [1, 5], pool_size = [1, 13], padding = [0, 6])
        )
        self.attn_block2 = SimpleAttentionalGain(10, 256, global_avg_cue=global_avg_cue)

        self.conv3 =  nn.Sequential(
            nn.LayerNorm([256, 10, 250]),
            conv2d_same.create_conv2d_pad(256, 512, kernel_size = [5, 5], stride = [1,1], padding = 'same'),
            nn.ReLU(),
            HannPooling2d(stride = [1, 4], pool_size = [1, 13], padding = [0, 6])
        )
        self.attn_block3 = SimpleAttentionalGain(10, 512, global_avg_cue=global_avg_cue)

        self.conv4 = nn.Sequential(
            nn.LayerNorm([512, 10, 63]),
            conv2d_same.create_conv2d_pad(512, 512, kernel_size = [6, 6], stride = [1,1], padding = 'same'),
            nn.ReLU(),
            HannPooling2d(stride = [1, 1], pool_size = [1, 1], padding = [0, 0])
        )
        self.attn_block4 = SimpleAttentionalGain(10, 512, global_avg_cue=global_avg_cue)

        self.conv5 = nn.Sequential(
            nn.LayerNorm([512, 10, 63]),
            conv2d_same.create_conv2d_pad(512, 512, kernel_size = [5, 5], stride = [1,1], padding = 'same'),
            nn.ReLU(),
            HannPooling2d(stride = [1, 1], pool_size = [1, 1], padding = [0, 0])
        )
        self.attn_block5 = SimpleAttentionalGain(10, 512, global_avg_cue=global_avg_cue)

        self.conv6 = nn.Sequential(
            nn.LayerNorm([512, 10, 63]),
            conv2d_same.create_conv2d_pad(512, 512, kernel_size = [6, 6], stride = [1,1], padding = 'same'),
            nn.ReLU(),
            HannPooling2d(stride = [2, 4], pool_size = [6, 13], padding = [3, 6])
        )
        self.attn_block6 = SimpleAttentionalGain(6, 512, global_avg_cue=global_avg_cue)

        self.final_conv_size = 512*6*16 # add fn to dynamically compute
        self.fullyconnected = nn.Linear(self.final_conv_size, fc_size)
        self.relufc = nn.ReLU()
        self.dropout = nn.Dropout()
            
        if self.dual_task:
            self.classificationLoc = nn.Linear(fc_size, num_Loc)
            self.classificationWord = nn.Linear(fc_size. num_word)
        else:
            self.classification = nn.Linear(fc_size, num_classes)
            

    def forward(self, cue=None, mixture=None, cue_mask_ixs=None):
        if cue == None:
            mixture = self.norm_coch_rep(mixture)
            mixture = self.conv0(mixture) 
            mixture = self.conv1(mixture)
            mixture = self.conv2(mixture)
            mixture = self.conv3(mixture)
            mixture = self.conv4(mixture)
            mixture = self.conv5(mixture)
            out = self.conv6(mixture)

        else:
            # pass cue through cnn & store reps
            cue = self.norm_coch_rep(cue)
            cue0 = self.conv0(cue) 
            cue1 = self.conv1(cue0)
            cue2 = self.conv2(cue1)
            cue3 = self.conv3(cue2)
            cue4 = self.conv4(cue3)
            cue5 = self.conv5(cue4)
            cue6 = self.conv6(cue5)

            ## Combine cue and mixture using attention
            # norm for first attn layer
            mixture = self.norm_coch_rep(mixture)
            # attn for cochlear model
            attn = self.attn_block_in(cue, mixture, cue_mask_ixs)
            # conv 0 
            attn = self.conv0(attn)
            attn = self.attn_block0(cue0, attn, cue_mask_ixs)
            # conv 1
            attn = self.conv1(attn)
            attn = self.attn_block1(cue1, attn, cue_mask_ixs)
            #conv 2
            attn = self.conv2(attn)
            attn = self.attn_block2(cue2, attn, cue_mask_ixs)
            #conv 3
            attn = self.conv3(attn)
            attn = self.attn_block3(cue3, attn, cue_mask_ixs)
            #conv4
            attn = self.conv4(attn)
            attn = self.attn_block4(cue4, attn, cue_mask_ixs)
            #conv5
            attn = self.conv5(attn)
            attn = self.attn_block5(cue5, attn, cue_mask_ixs)
            #conv6
            attn = self.conv6(attn)
            out = self.attn_block6(cue6, attn, cue_mask_ixs)

        out = out.view(out.size(0), self.final_conv_size) # B x conv feature size
        out = self.fullyconnected(out)        
        out = self.relufc(out)
        out = self.dropout(out)        
        if self.dual_task:
            word_out = self.classificationWord(out)
            loc_out = self.classificationLoc(out)
            return word_out, loc_out
        else:
            return self.classification(out)