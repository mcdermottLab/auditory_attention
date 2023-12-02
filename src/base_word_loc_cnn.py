import torch
import torch.nn as nn
from src.layers import conv2d_same
from src.custom_modules import HannPooling2d


class AuditoryCNN(nn.Module):
    def __init__(self, num_words=800, num_locs=512, fc_size=512, dropout=0.5):
        super(AuditoryCNN, self).__init__()

        self.conv0 = nn.Sequential(
                    nn.LayerNorm([2, 40, 20000]),
                    conv2d_same.create_conv2d_pad(2, 32, kernel_size = [2, 34], stride = [1, 1], padding = 'same'),
                    nn.ReLU(),
                    HannPooling2d(stride = [2, 4], pool_size = [9, 13], padding = [4, 6])
        )

        self.conv1 = nn.Sequential(
            nn.LayerNorm([32, 20, 5000]),
            conv2d_same.create_conv2d_pad(32, 64, kernel_size = [2, 14], stride = [1,1], padding = 'same'),
            nn.ReLU(),
            HannPooling2d(stride = [2, 4], pool_size = [9, 13], padding = [4, 6])
        )

        self.conv2 = nn.Sequential(
            nn.LayerNorm([64, 10, 1250]),
            conv2d_same.create_conv2d_pad(64, 256, kernel_size = [5, 5], stride = [1,1], padding = 'same'),
            nn.ReLU(),
            HannPooling2d(stride = [1, 5], pool_size = [1, 13], padding = [0, 6])
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

        self.fc_norm = nn.LayerNorm([512,6,16]) 
        self.word_fc = nn.Linear(512*6*16, fc_size)
        self.word_relufc = nn.ReLU()
        self.word_dropout = nn.Dropout(dropout)
        self.word_classification = nn.Linear(fc_size, num_words)

        self.loc_fc = nn.Linear(512*6*16, fc_size)
        self.loc_relufc = nn.ReLU()
        self.loc_dropout = nn.Dropout(dropout) 
        self.loc_classification = nn.Linear(fc_size, num_locs)  
    
        

    def forward(self, x, *args):
        # pass xthrough cnn & store reps
        # x = self.norm_coch_rep(x)
        x = self.conv0(x) # has layer norm as 1st layer - may be a problem? 
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)

        x = self.fc_norm(x)
        x = x.view(x.size(0),-1) # B x FC size

        # word classification
        word_x = self.word_fc(x)
        word_x = self.word_relufc(word_x)
        word_x = self.word_dropout(word_x)  
        word_x = self.word_classification(word_x)

        # location classification
        loc_x = self.loc_fc(x)
        loc_x = self.loc_relufc(loc_x)
        loc_x = self.loc_dropout(loc_x)
        loc_x = self.loc_classification(loc_x)

        return word_x, loc_x
        
    