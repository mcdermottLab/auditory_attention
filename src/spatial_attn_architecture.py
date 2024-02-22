import torch
import numpy as np
import torch.nn as nn
from collections import OrderedDict
from src.layers import conv2d_same
from src.layers import padding as pad_utils
from src.custom_modules import HannPooling2d

class SimpleAttentionalGain(nn.Module):
    def __init__(self, frequency_dim, cnn_channels, global_avg_cue=False, n_cue_frames=None, **kwargs):
        super(SimpleAttentionalGain, self).__init__()
        self.frequency_dim = frequency_dim
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
    
        self.reset_parameters() 

    def reset_parameters(self):
        nn.init.constant_(self.bias, 0)
        nn.init.constant_(self.slope, 1)
        nn.init.constant_(self.threshold, 0)

    def forward(self, cue, mixture, cue_mask_ixs):
        ## Process cue 
        # time average - same as nn op, but is compat with compile 
        # if self.n_cue_frames:
        #     cue_dur = cue.shape[-1]
        #     diff = (cue_dur - self.n_cue_frames) // 2
        #     frame_start = diff 
        #     frame_end = int(cue_dur - diff)
        #     cue = cue[..., frame_start : frame_end]
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
        mixture = torch.mul(mixture, gain)
        return mixture


class CNN2DExtractor(nn.Module):
    ''' CNN wrapper, includes relu and layer-norm if applied'''

    def __init__(self, input_sr, out_channels, kernel, stride, padding, pool_stride, pool_size, pool_padding, attn, dropout,
                  fc_size=512, global_avg_cue=False, num_classes={"num_words":998, "num_locs":504}, residual_attn=False, double_size=False, n_cue_frames=None, **kwargs):
        super(CNN2DExtractor, self).__init__()
        # Setup
        print(f"{num_classes=}")
        self.dual_task = False
        if isinstance(num_classes, dict):
            class_keys = num_classes.keys()
            if ("num_words" in class_keys) and not ("num_locs" in class_keys):
                # only_word 
                num_classes = num_classes['num_words']
                # print('Model performing word task')
            elif ("num_locs" in class_keys) and not ("num_words" in class_keys):
                # only_loc
                num_classes = num_classes['num_locs']
                # print('Model performing location task')

            elif ("num_locs" in class_keys) and ("num_words" in class_keys):
                self.dual_task = True
                num_words = num_classes['num_words']
                num_locs = num_classes['num_locs']
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
        self.fullyconnected = nn.Linear(self.output_size, fc_size)
        self.relufc = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        if self.dual_task:
            self.classificationWord = nn.Linear(fc_size, num_words)
            self.classificationLoc = nn.Linear(fc_size, num_locs)
        else:
            self.classification = nn.Linear(fc_size, num_classes)

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


class BinauralAuditoryAttentionCNN(nn.Module):

    def __init__(self, input_sr, out_channels, kernel, stride, padding, pool_stride, pool_size, pool_padding, attn, dropout,
                  fc_size=512, global_avg_cue=False, num_classes={"num_words":800, "num_locs":504}, frequency_dim=40,
                  residual_attn=False, n_cue_frames=None, starting_output_len = 20000, norm_first=True, ln_affine=True, **kwargs):
        super(BinauralAuditoryAttentionCNN, self).__init__()
        # Setup
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

        # build architecture
        self.model_dict["norm_coch_rep"]= nn.LayerNorm([self.input_channels, self.frequency_dim, self.output_len])
        # self.model_dict["attn_block_in"] = SimpleAttentionalGain(self.frequency_dim, self.input_channels, global_avg_cue=global_avg_cue, n_cue_frames=self.n_cue_frames)

        for idx in range(self.n_layers):
        
            # print(f"output height: {self.output_height}, output len: {self.output_len}")

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
                pass
            else:
                conv_padding, _ = pad_utils.get_padding_value(self.padding[idx], self.kernel[idx], stride=self.stride[idx])
                output_height = int(np.floor((self.output_height + (2 * conv_padding[0]) - (kernel[idx][0] - 1) - 1) / stride[idx][0]) + 1)
                output_len = int(np.floor((self.output_len + (2 * conv_padding[1]) -  (kernel[idx][1] - 1) - 1) / stride[idx][1]) + 1)
                if self.n_cue_frames:
                    n_cue_frames =  int(np.floor((self.n_cue_frames + (2 * conv_padding[1]) -  (kernel[idx][1] - 1) - 1) / stride[idx][1]) + 1)


            if self.norm_first:
                print(f'nIn: {nIn}, nOut: {nOut}, kernel: {self.kernel[idx]}, stride: {self.stride[idx]}, padding: {self.padding[idx]}')
                # if norm before conv, can use prior output shapes for norm layer
                print(f"output height: {self.output_height}, output len: {self.output_len}")

                block = nn.Sequential(nn.LayerNorm([nIn, self.output_height, self.output_len], elementwise_affine=self.ln_affine),
                                    conv2d_same.create_conv2d_pad(nIn, nOut, self.kernel[idx], stride=self.stride[idx], padding=self.padding[idx]),
                                    nn.ReLU())
            else:  
                # if norm after conv, use new output shapes for norm layer
                print(f'nIn: {nIn}, nOut: {nOut}, kernel: {self.kernel[idx]}, stride: {self.stride[idx]}, padding: {self.padding[idx]}')
                print(f"output height: {output_height}, output len: {output_len}")
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
        self.fullyconnected = nn.Linear(self.output_size, fc_size)
        self.relufc = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        if self.dual_task:
            self.classificationWord = nn.Linear(fc_size, num_words)
            self.classificationLoc = nn.Linear(fc_size, num_locs)
        else:
            self.classification = nn.Linear(fc_size, num_classes)

    def forward(self, cue=None, mixture=None, cue_mask_ixs=None):
        if cue == None:
            mixture = self.model_dict["norm_coch_rep"](mixture)
            for idx in range(self.n_layers):
                mixture = self.model_dict[f'conv_block_{idx}'](mixture)
                # print(f"conv_block_{idx}, {mixture.max(), mixture.min()}")
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
        out = self.fullyconnected(out)        
        out = self.relufc(out)
        out = self.dropout(out)        
        if self.dual_task:
            word_out = self.classificationWord(out)
            loc_out = self.classificationLoc(out)
            return word_out, loc_out
        else:
            return self.classification(out)




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
