import os
import sys
import pdb
import numpy as np
import torch
import scipy.signal

def freq2erb(freq):
    """
    Helper function converts frequency from Hz to ERB-number scale.
    Glasberg & Moore (1990, Hearing Research) equation 4. The ERB-
    number scale can be defined as the number of equivalent
    rectangular bandwidths below the given frequency (units of the
    ERB-number scale are Cams).
    """
    return 21.4 * np.log10(0.00437 * freq + 1.0)


def erb2freq(erb):
    """
    Helper function converts frequency from ERB-number scale to Hz.
    Glasberg & Moore (1990, Hearing Research) equation 4. The ERB-
    number scale can be defined as the number of equivalent
    rectangular bandwidths below the given frequency (units of the
    ERB-number scale are Cams).
    """
    return (1.0 / 0.00437) * (np.power(10.0, (erb / 21.4)) - 1.0)


def erbspace(freq_min, freq_max, num):
    """
    Helper function to get array of frequencies linearly spaced on an
    ERB-number scale.
    
    Args
    ----
    freq_min (float): minimum frequency in Hz
    freq_max (float): maximum frequency Hz
    num (int): number of frequencies (length of array)
    
    Returns
    -------
    freqs (np.ndarray): array of ERB-spaced frequencies (lowest to highest) in Hz
    """
    freqs = np.linspace(freq2erb(freq_min), freq2erb(freq_max), num=num)
    freqs = erb2freq(freqs)
    return freqs


def get_half_cosine_transfer_function(lo=None, hi=None, cf=None, bw_erb=None):
    """
    Returns the transfer function for a half-cosine filter, which
    can be specified by either a low and high frequency cutoff or
    a center frequency (in Hz) and bandwidth (in ERB).
    """
    msg = "Specify half cosine filter using either (lo, hi) XOR (cf, bw_erb)"
    if (lo is not None) and (hi is not None):
        assert (cf is None) and (bw_erb is None), msg
        lo_erb = freq2erb(lo)
        hi_erb = freq2erb(hi)
        cf_erb = (lo_erb + hi_erb) / 2
        bw_erb = hi_erb - lo_erb
    elif (cf is not None) and (bw_erb is not None):
        assert (lo is None) and (hi is None), msg
        cf_erb = freq2erb(cf)
        lo_erb = cf_erb - (bw_erb / 2)
        hi_erb = cf_erb + (bw_erb / 2)
    else:
        raise ValueError(msg)
    
    def half_cosine_transfer_function(f):
        f_erb = freq2erb(f)
        out = np.zeros_like(f_erb)
        IDX = np.logical_and(f_erb > lo_erb, f_erb < hi_erb)
        out[IDX] = np.cos(np.pi * (f_erb[IDX] - cf_erb) / bw_erb)
        return out
    return half_cosine_transfer_function


def make_half_cosine_filters(signal_length,
                             sr,
                             list_lo=None,
                             list_hi=None):
    """
    Function builds a filterbank of half-cosine bandpass filters.
    
    Args
    ----
    signal_length (int): length of signal (in samples) that filters will be applied to
    sr (int): sampling rate (Hz)
    list_lo (np.ndarray): filterbank low frequency cutoffs (Hz)
    list_hi (np.ndarray): filterbank high frequency cutoffs (Hz)
    
    Returns
    -------
    filts (np.ndarray): half-cosine filterbank, an array of floats with shape [num_cf, num_freq]
    freqs (np.ndarray): frequency vector (Hz)
    """
    assert (list_lo is not None) and (list_hi is not None), "list_lo and list_hi are required args"
    assert list_lo.shape == list_hi.shape, "list_lo and list_hi must be arrays with the same shape"
    num_cf = list_lo.shape[0]
    # Setup frequency vector and initialize filter array
    if np.remainder(signal_length, 2) == 0: # even length
        num_freq = int(signal_length // 2) + 1
        max_freq = sr / 2
    else: # odd length
        num_freq = int((signal_length - 1) // 2) + 1
        max_freq = sr * (signal_length - 1) / 2 / signal_length
    freqs = np.linspace(0, max_freq, num_freq)
    filts = np.zeros((num_cf, num_freq))
    # Build the half-cosine filterbank
    for fidx, (lo, hi) in enumerate(zip(list_lo, list_hi)):
        assert lo < hi, "low frequency cutoff must be < high frequency cutoff"
        halfcos = get_half_cosine_transfer_function(lo=lo, hi=hi)
        filts[fidx, :] = halfcos(freqs)
    return filts, freqs


def half_cosine_filterbank(tensor_input,
                           sr,
                           num_cf=50,
                           min_lo=20.0,
                           max_hi=10e3,
                           return_io_function=True):
    """
    """
    # Prepare list of CFs / BWs (default is spaced linearly on an ERB scale)
    list_cutoffs = erbspace(min_lo, max_hi, num_cf + 2)
    list_lo = list_cutoffs[:-2]
    list_hi = list_cutoffs[2:]
    cfs = list_cutoffs[1:-1]
    bws = list_hi - list_lo
    bws_erb = freq2erb(list_hi) - freq2erb(list_lo)
    # Prepare input signal and apply half-cosine filterbank in frequency domain
    filts, freqs = make_half_cosine_filters(
        tensor_input.shape[-1],
        sr,
        list_lo=list_lo,
        list_hi=list_hi)
    rfft_filts = torch.unsqueeze(torch.tensor(filts, dtype=torch.complex64),0)
    
    def filterbank_io_function(x):
        # Prepare input signal and apply roex filterbank in frequency domain
        if len(x.shape) == 1:
            x = torch.unsqueeze(x,0)
        elif len(x.shape) > 2:
            raise ValueError("Input dimensions should be: [batch, time]")
        rfft_x = torch.fft.rfft(x.to(torch.float32))
        rfft_x = torch.unsqueeze(rfft_x,1)
        rfft_y = torch.multiply(rfft_x, rfft_filts)
        y = torch.fft.irfft(rfft_y)
        print(y.shape)
        return y
    
    if return_io_function:
        return filterbank_io_function
    else:
        return filterbank_io_function(tensor_input)


def tf_fir_resample(tensor_input,
                    sr_input,
                    sr_output,
                    kwargs_fir_lowpass_filter={},
                    verbose=True,
                    return_io_function=True):
    """
    Tensorflow function for resampling time-domain signals with an FIR lowpass filter.
    
    Args
    ----
    tensor_input (tensor): input tensor to be resampled along time dimension (expects shape
        [batch, time], [batch, freq, time], or [batch, freq, time, 1])
    sr_input (int): input sampling rate in Hz
    sr_output (int): output sampling rate in Hz
    kwargs_fir_lowpass_filter (dict): keyword arguments for fir_lowpass_filter,
        which can be used to alter cutoff frequency of anti-aliasing lowpass filter
    verbose (bool): if True, function will print optional information
    return_io_function (bool): if True, function will return a wrapper function that
        applies resampling operation. Otherwise, function will return resampled tensor
    
    Returns
    -------
    resample_io_function (function): wrapper function maps tensors to resampled tensors
    < OR >
    tensor_input_resampled (tensor): resampled tensor with shape matched to tensor_input
    """
    # Expand dimensions of input tensor to [batch, freq, time, channels] for 2d conv operation
    if len(tensor_input.shape) == 2:
        f = 1
        t = tensor_input.shape[1]
        c = 1
        tensor_input = torch.unsqueeze(torch.unsqueeze(tensor_input,1),-1)
        if verbose:
            print(f'[tf_fir_resample] interpreted `tensor_input.shape` as [batch, time={t}]')
    elif len(tensor_input.shape) == 3:
        f = tensor_input.shape[1]
        t = tensor_input.shape[2]
        c = 1
        tensor_input = torch.unsqueeze(tensor_input,-1)
        if verbose:
            print(f'[tf_fir_resample] interpreted `tensor_input.shape` as [batch, freq={f}, time={t}]')
    else:
        msg = "dimensions of `tensor_input` must support re-shaping to [batch, freq, time, channels]"
        assert (len(tensor_input.shape) == 4), msg
        f = tensor_input.shape[1]
        t = tensor_input.shape[2]
        c = tensor_input.shape[3]
        tensor_input = tensor_input
        if verbose:
            print(f'[tf_fir_resample] interpreted `tensor_input.shape` as [batch, freq={f}, time={t}, channels={c}]')
    # Compute upsample and downsample factors
    greatest_common_divisor = np.gcd(int(sr_output), int(sr_input))
    up = int(sr_output) // greatest_common_divisor
    down = int(sr_input) // greatest_common_divisor
    # First upsample by a factor of `up` by adding `up-1` zeros between each sample in original signal
    nzeros = up - 1
    if nzeros > 0:
        paddings = (0,0,0,1,0,0,0,0)
        indices = []
        for _ in range(t):
            indices.append(_)
            indices.extend([t] * nzeros)
        tensor_input_padded = torch.nn.functional.pad(
            tensor_input,
            pad=paddings,
            mode='constant')
        # Insert up - 1 zeros between each sample
        tensor_input_padded = tensor_input_padded[:, :, indices]
    else:
        tensor_input_padded = tensor_input

    # Next construct anti-aliasing lowpass filter
    kwargs_fir_lowpass_filter = dict(kwargs_fir_lowpass_filter) # prevents modifying in-place
    if kwargs_fir_lowpass_filter.get('cutoff', None) is None:
        kwargs_fir_lowpass_filter['cutoff'] = sr_output / 2
    if verbose:
        print('[tf_fir_resample] `kwargs_fir_lowpass_filter`: {}'.format(kwargs_fir_lowpass_filter))
    
    filt, sr_filt = fir_lowpass_filter(sr_input, sr_output, **kwargs_fir_lowpass_filter, verbose=verbose)
    filt = filt * up # Re-scale filter to offset attenuation from upsampling
    filt = torch.unsqueeze(torch.unsqueeze(torch.from_numpy(filt),0),0).float()
    tensor_kernel_lowpass_filter = filt

    pad_length = int((tensor_kernel_lowpass_filter.shape[-1]-1)/2)
    paddings = (pad_length, pad_length)
    tensor_output = torch.nn.functional.pad(
        tensor_input_padded.squeeze(-1),
        pad=paddings,
        mode='constant') 

    tensor_input_lowpass_filtered = torch.nn.functional.conv2d(tensor_output.unsqueeze(1),tensor_kernel_lowpass_filter.unsqueeze(1),stride=(1,down), padding='valid')

    tensor_output = tensor_input_lowpass_filtered.squeeze(1)

    return tensor_output
    

def fir_lowpass_filter(sr_input,
                       sr_output,
                       numtaps=None,
                       dur_fir=None,
                       cutoff=None,
                       window=('kaiser', 5.0),
                       verbose=False):
    """
    Build an anti-aliasing finite impulse response lowpass filter.
    
    Args
    ----
    sr_input (int): input sampling rate (Hz)
    sr_output (int): output sampling rate (Hz)
    numtaps (int): length of FIR in samples at filter sampling rate
    dur_fir (float): duration of finite impulse response in seconds
    cutoff (int): lowpass filter cutoff frequency in Hz
    window (tuple): argument for `scipy.signal.windows.get_window`
    
    Returns
    -------
    filt (np.ndarray): finite impulse response of lowpass filter
    sr_filt (int): sampling rate of lowpass filter impulse response
    """
    none_args = [_ for _ in [numtaps, dur_fir] if _ is None]
    assert len(none_args) == 1, "Specify exactly one of [numtaps, dur_fir]"
    if sr_output is None:
        sr_output = sr_input
    greatest_common_divisor = np.gcd(int(sr_input), int(sr_output))
    down = int(sr_input) // greatest_common_divisor
    up = int(sr_output) // greatest_common_divisor
    sr_filt = sr_input * up
    if cutoff is None:
        cutoff = sr_output / 2
    if dur_fir is None:
        dur_fir = int(2 * (numtaps // 2)) / sr_filt
    else:
        numtaps = int(2 * (dur_fir * sr_filt // 2)) + 1 # Ensure numtaps from dur_fir is odd
    assert cutoff <= sr_output / 2, "cutoff may not exceed Nyquist"
    filt = scipy.signal.firwin(
        numtaps=numtaps,
        cutoff=cutoff,
        width=None,
        window=tuple(window),
        pass_zero=True,
        scale=True,
        fs=sr_filt)
    if verbose:
        print("[fir_lowpass_filter] sr_filt = {} Hz".format(sr_filt))
        print("[fir_lowpass_filter] numtaps = {} samples".format(numtaps))
        print("[fir_lowpass_filter] dur_fir = {} seconds".format(dur_fir))
        print("[fir_lowpass_filter] cutoff = {} Hz".format(cutoff))
        print("[fir_lowpass_filter] window = {}".format(tuple(window)))
    return filt, sr_filt