import sys
sys.path.append('/om/user/imgriff/python-packages/')
import chcochleagram
import chcochleagram.compression as compression
import chcochleagram.downsampling as downsampling
import torch as ch
import numpy as np
import scipy.io as sio


class TimeDomainCochleagram(ch.nn.Module):
    """
    Generates a cochleagram, passing an audio signal through a set of bandpass
    filters designed to mimic an ear, followed by envelope extraction, downsampling,
    and a compressive nonlinearity. Filters applied using convolution in the time
    domain.

    """
    def __init__(self, filter_params, downsampling, compression=None, use_pad=None):
        """
        Makes the torch components used for the cochleagram generation.

        Args:
            filter_object (cochlear_filters.CochlearFilter) : a set of torch
                filters that will be used for the cochleagram generation. Many
                parameters for cochleagram generation are inherited from the
                filters that are constructed.
            envelope_extraction (envelope_extraction.EnvelopeExtraction) :
        """
        super(TimeDomainCochleagram, self).__init__()
        self.path_to_filters = filter_params['coch_filter_path']
        self.numpy_coch_filters = sio.loadmat(self.path_to_filters)['IR']
        self.numpy_coch_filters = np.ascontiguousarray(np.fliplr(self.numpy_coch_filters)) # flip if using torch conv
        self.use_pad = use_pad
        self.compute_subbands = ComputeSubbands(self.numpy_coch_filters,
                                                self.use_pad)
        self.downsampling = downsampling
        self.compression = compression

    def forward(self, x):
        x = self.compute_subbands(x)
        x = ch.nn.functional.relu(x, inplace=True)
        x = self.downsampling(x)
        if self.compression is not None:
            x = self.compression(x)
        return x

class ComputeSubbands(ch.nn.Module):
    """
    Convolves input with impulse response of filters in cochlear filter
    bank.
    """
    def __init__(self, coch_filters, use_pad):
        super(ComputeSubbands, self).__init__()
        self.n_taps = coch_filters.shape[1]
        if use_pad:
            pad_factor = self.n_taps - 1 # need odd number for total len
            coch_filters = np.pad(coch_filters, ((0,0), (pad_factor,0)), mode='constant', constant_values=0)
        self.n_channels = coch_filters.shape[0]
        coch_filters = coch_filters.reshape(self.n_channels,
                                            1,
                                            -1)
        self.register_buffer("coch_filters", ch.from_numpy(coch_filters).float())

    def forward(self, x):
        x_shape = x.shape
        if len(x_shape)>2:
            x = x.view(x_shape[0]*x_shape[-2], 1, -1)
        else: # Handle the case where there is no batch dimension
            x = x.view(x_shape[0], 1, -1)
        x = ch.nn.functional.conv1d(x, self.coch_filters, padding='same')
        x = x.view(x_shape[0], 1, self.n_channels, -1)
        return x
