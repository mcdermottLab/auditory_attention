import torch as ch
from torch import nn
from src import audio_transforms
from torch._jit_internal import _copy_to_script_wrapper
import numpy as np
from collections import OrderedDict


class FakeReLU(ch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

class FakeReLUM(nn.Module):
    def forward(self, x):
        return FakeReLU.apply(x)


class HannPooling2d(nn.Module):
    """
    2D Weighted average pooling with a Hann window.
    
    Inputs
    ------
    stride (int): amount of subsampling
    pool_size (int or list): width of the hann window (if int, assume window is square)
    padding (string): type of padding for the convolution
    normalize (bool): if true, divide the filter by the sum of its values, so that the 
        smoothed signal is the same amplitude as the original.
    """
    def __init__(self, stride, pool_size, padding=0, normalize=True):
        super(HannPooling2d, self).__init__()
        self.stride = stride
        if type(pool_size)==int:
            pool_size = [pool_size, pool_size]
        self.pool_size = pool_size
        self.padding = padding
        self.normalize = normalize
        
        hann_window2d = self._make_hann_window()
        self.register_buffer('hann_window2d', ch.from_numpy(hann_window2d).float())
        
    def forward(self, x): # TODO: implement different padding
        # TODO: is this the fastest way to apply the weighted average?
        # https://discuss.pytorch.org/t/applying-conv2d-filter-to-all-channels-seperately-is-my-solution-efficient/22840/2
        x_shape = x.shape
        if len(x_shape)==2: # Assume no batch or channel dimension
            h,w = x_shape
            x = x.view(1,1,h,w)
        elif len(x_shape)==3: # Assume no batch dimension
            c,h,w = x_shape
            x = x.view(c, 1, h, w)
        elif len(x_shape)==4:
            b,c,h,w = x_shape
            x = x.view(b*c, 1, h, w)
        x = ch.nn.functional.conv2d(x, self.hann_window2d, 
                                    stride=self.stride, 
                                    padding=self.padding)
        x_shape_after_filt = x.shape
        return x.view(x_shape[0:-2] + (x_shape_after_filt[-2],x_shape_after_filt[-1]))
    
    def _make_hann_window(self):
        hann_window_h = np.expand_dims(np.hanning(self.pool_size[0]),0)
        hann_window_w = np.expand_dims(np.hanning(self.pool_size[1]),1)
        
        # Add a channel dimensiom to the filter
        hann_window2d = np.expand_dims(np.expand_dims(np.outer(hann_window_h, hann_window_w),0),0)
        
        if self.normalize:
            hann_window2d = hann_window2d/(sum(hann_window2d.ravel()))
        return hann_window2d


class HannPooling1d(nn.Module):
    """
    1D Weighted average pooling with a Hann window.
    
    Inputs
    ------
    stride (int): amount of subsampling
    pool_size (int): width of the hann window
    padding (string): type of padding for the convolution
    normalize (bool): if true, divide the filter by the sum of its values, so that the 
        smoothed signal is the same amplitude as the original.
    """
    def __init__(self, stride, pool_size, padding=0, normalize=True):
        super(HannPooling1d, self).__init__()
        self.stride = stride
        self.pool_size = pool_size
        self.padding = padding
        self.normalize = normalize
        
        hann_window1d = self._make_hann_window()
        self.register_buffer('hann_window1d', ch.from_numpy(hann_window1d).float())
        
    def forward(self, x): # TODO: implement different padding
        # TODO: is this the fastest way to apply the weighted average?
        # https://discuss.pytorch.org/t/applying-conv2d-filter-to-all-channels-seperately-is-my-solution-efficient/22840/2
        x_shape = x.shape
        if len(x_shape)==1: # Assume no batch or channel dimension
            w = x_shape[0]
            x = x.view(1,1,w)
        elif len(x_shape)==2: # Assume no batch dimension
            c,w = x_shape
            x = x.view(c, 1, w)
        elif len(x_shape)==3:
            b,c,w = x_shape
            x = x.view(b*c, 1, w)
        x = ch.nn.functional.conv1d(x, self.hann_window1d,
                                    stride=self.stride, 
                                    padding=self.padding)
        x_shape_after_filt = x.shape
        return x.view(x_shape[0:-1] + (x_shape_after_filt[-1],))
    
    def _make_hann_window(self):
        hann_window_w = np.hanning(self.pool_size)
        
        # Add a channel dimensiom to the filter
        hann_window1d = np.expand_dims(np.expand_dims(hann_window_w,0),0)
        
        if self.normalize:
            hann_window1d = hann_window1d/(sum(hann_window1d.ravel()))
        return hann_window1d


class SequentialWithArgs(ch.nn.Sequential):
    def forward(self, input, *args, **kwargs):
        vs = list(self._modules.values())
        l = len(vs)
        for i in range(l):
            if i == l-1:
                input = vs[i](input, *args, **kwargs)
            else:
                input = vs[i](input)
        return input

class AudioInputRepresentation(ch.nn.Module):
    '''
    A module (custom layer) for turning the audio signal into a
    representation for training, ie using a mel spectrogram or a
    cochleagram.
    '''
    def __init__(self, rep_type, rep_kwargs, compression_type, compression_kwargs):
        super(AudioInputRepresentation, self).__init__()
        self.rep_type = rep_type
        self.rep_kwargs = rep_kwargs
        self.compression_type = compression_type
        self.compression_kwargs = compression_kwargs

        # Functions for the representations are defined in the audio_transforms
        # library, but we only use the foreground audio here.
        self.full_rep = audio_transforms.AudioToAudioRepresentation(rep_type,
                                                                    rep_kwargs,
                                                                    compression_type,
                                                                    compression_kwargs)

    def forward(self, x, output_lens=None): 
        # print(self.full_rep)
        x, _ = self.full_rep(x, None)
        return x

class AttnAudioInputRepresentation(ch.nn.Module):
    '''
    A module (custom layer) for turning the audio signal into a
    representation for training, ie using a mel spectrogram or a
    cochleagram.
    '''
    def __init__(self, rep_type, rep_kwargs, compression_type, compression_kwargs, **kwargs):
        super(AttnAudioInputRepresentation, self).__init__()
        self.rep_type = rep_type
        self.rep_kwargs = rep_kwargs
        self.compression_type = compression_type
        self.compression_kwargs = compression_kwargs

        # Functions for the representations are defined in the audio_transforms
        # library, but we only use the foreground audio here.
        self.full_rep = audio_transforms.AudioToAudioRepresentation(rep_type,
                                                                    rep_kwargs,
                                                                    compression_type,
                                                                    compression_kwargs)

    def forward(self, cue, mixture, output_lens=None): 
        # print(self.full_rep)
        cue, _ = self.full_rep(cue, None)
        mixture, _ = self.full_rep(mixture, None)
        return cue, mixture

class SequentialAttacker(ch.nn.Module):
    r"""A sequential container with additional kwargs for attacker models.
    Based on ch.nn.Sequential
    Modules will be added to it in the order they are passed in the constructor.
    Alternatively, an ordered dict of modules can also be passed in.

    To make it easier to understand, here is a small example::

        # Example of using Sequential
        model = nn.Sequential(
                  nn.Conv2d(1,20,5),
                  nn.ReLU(),
                  nn.Conv2d(20,64,5),
                  nn.ReLU()
                )

        # Example of using Sequential with OrderedDict
        model = nn.Sequential(OrderedDict([
                  ('conv1', nn.Conv2d(1,20,5)),
                  ('relu1', nn.ReLU()),
                  ('conv2', nn.Conv2d(20,64,5)),
                  ('relu2', nn.ReLU())
                ]))
    """

    def __init__(self, *args):
        super(SequentialAttacker, self).__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)

    def _get_item_by_idx(self, iterator, idx):
        """Get the idx-th item of the iterator"""
        size = len(self)
        idx = operator.index(idx)
        if not -size <= idx < size:
            raise IndexError('index {} is out of range'.format(idx))
        idx %= size
        return next(islice(iterator, idx, None))

    @_copy_to_script_wrapper
    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return self.__class__(OrderedDict(list(self._modules.items())[idx]))
        else:
            return self._get_item_by_idx(self._modules.values(), idx)

    def __setitem__(self, idx, module):
        key = self._get_item_by_idx(self._modules.keys(), idx)
        return setattr(self, key, module)

    def __delitem__(self, idx):
        if isinstance(idx, slice):
            for key in list(self._modules.keys())[idx]:
                delattr(self, key)
        else:
            key = self._get_item_by_idx(self._modules.keys(), idx)
            delattr(self, key)

    @_copy_to_script_wrapper
    def __len__(self):
        return len(self._modules)

    @_copy_to_script_wrapper
    def __dir__(self):
        keys = super(Sequential, self).__dir__()
        keys = [key for key in keys if not key.isdigit()]
        return keys

    @_copy_to_script_wrapper
    def __iter__(self):
        return iter(self._modules.values())

    def forward(self, input, *args, **kwargs):
        for module in self:
            input = module(input, *args, **kwargs)
        return input


class AttnSequentialAttacker(ch.nn.Module):
    r"""A sequential container with additional kwargs for attacker models.
    Based on ch.nn.Sequential
    Modules will be added to it in the order they are passed in the constructor.
    Alternatively, an ordered dict of modules can also be passed in.

    To make it easier to understand, here is a small example::

        # Example of using Sequential
        model = nn.Sequential(
                  nn.Conv2d(1,20,5),
                  nn.ReLU(),
                  nn.Conv2d(20,64,5),
                  nn.ReLU()
                )

        # Example of using Sequential with OrderedDict
        model = nn.Sequential(OrderedDict([
                  ('conv1', nn.Conv2d(1,20,5)),
                  ('relu1', nn.ReLU()),
                  ('conv2', nn.Conv2d(20,64,5)),
                  ('relu2', nn.ReLU())
                ]))
    """

    def __init__(self, *args):
        super(AttnSequentialAttacker, self).__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)

    def _get_item_by_idx(self, iterator, idx):
        """Get the idx-th item of the iterator"""
        size = len(self)
        idx = operator.index(idx)
        if not -size <= idx < size:
            raise IndexError('index {} is out of range'.format(idx))
        idx %= size
        return next(islice(iterator, idx, None))

    @_copy_to_script_wrapper
    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return self.__class__(OrderedDict(list(self._modules.items())[idx]))
        else:
            return self._get_item_by_idx(self._modules.values(), idx)

    def __setitem__(self, idx, module):
        key = self._get_item_by_idx(self._modules.keys(), idx)
        return setattr(self, key, module)

    def __delitem__(self, idx):
        if isinstance(idx, slice):
            for key in list(self._modules.keys())[idx]:
                delattr(self, key)
        else:
            key = self._get_item_by_idx(self._modules.keys(), idx)
            delattr(self, key)

    @_copy_to_script_wrapper
    def __len__(self):
        return len(self._modules)

    @_copy_to_script_wrapper
    def __dir__(self):
        keys = super(Sequential, self).__dir__()
        keys = [key for key in keys if not key.isdigit()]
        return keys

    @_copy_to_script_wrapper
    def __iter__(self):
        return iter(self._modules.values())

    def forward(self, cue, mixture, *args, **kwargs):
        for module in self:
            outs = module(cue, mixture, *args, **kwargs)
            if isinstance(outs, tuple):
                cue, mixture = outs
        return outs
