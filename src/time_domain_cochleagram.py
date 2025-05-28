import sys
sys.path.append('/om/user/imgriff/python-packages/chcochleagram')
sys.path.append('/om/user/imgriff/python-packages/')
import chcochleagram
import chcochleagram.compression as compression
import chcochleagram.downsampling as downsampling
import torch
import numpy as np
import scipy.io as sio
import torchaudio.functional as F
import torchaudio.transforms as T
from scipy import signal as sig


class TimeDomainCochleagram(torch.nn.Module):
    """
    Generates a cochleagram, passing an audio signal through a set of bandpass
    filters designed to mimic an ear, followed by envelope extraction, downsampling,
    and a compressive nonlinearity. Filters applied using convolution in the time
    domain.

    """
    def __init__(self, filter_params, downsampling, compression=None,
                 use_pad=None, rep_on_gpu=False, binaural=False, impulse_len=1,
                 center_crop=False, out_dur=2, **kwargs):
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
        # make erb filter coefs
        self.erb_coefs, self.cf = make_ERB_filters(filter_params['sr'],
                                          filter_params['n_channels'],
                                          filter_params['low_lim'])
        self.erb_coefs = torch.from_numpy(self.erb_coefs).float()
        self.center_crop = center_crop
        self.n_out_frames = int(out_dur * filter_params['sr'])
        self.binaural = binaural
        # if rep_on_gpu - we'll pre-fab an impulse response for a convolutional FIR filter:
        if rep_on_gpu:
                window_size = 10
                impulse_dur = int(filter_params['sr'] * impulse_len)
                ir = torch.hstack([torch.ones(1,1), torch.zeros(1,impulse_dur - 1)])
                kernel = ERB_filter_bank(ir, self.erb_coefs)
                # window kernel
                kernel = backend_hann2d(kernel.numpy().squeeze(), window_size, filter_params['sr'])
                kernel = torch.from_numpy(kernel).float()
                kernel = torch.fliplr(kernel) # needed for conv layer to perform conv not auto. cor. 
                self.compute_rep = ComputeSubbands(kernel, use_pad)
                self.cat_dim = 1
        # if on cpu, we can use an IIR filter directly 
        else:
            self.compute_rep = lambda x: ERB_filter_bank(x, self.erb_coefs)
            self.cat_dim = 0

        self.downsampling = downsampling
        self.compression = compression

    def forward(self, x):
        if self.binaural:
            left_x = self.compute_rep(x[:, 0, :])
            right_x = self.compute_rep(x[:, 1, :])
            x = torch.cat([left_x, right_x], dim=self.cat_dim)
        else:
            x = self.compute_rep(x)
        if self.center_crop and x.shape[-1] > self.n_out_frames:
            x_dur = x.shape[-1]
            diff = (x_dur - self.n_out_frames) // 2
            frame_start = diff 
            frame_end = int(x_dur - diff)
            x = x[..., frame_start : frame_end]
        x = torch.nn.functional.relu(x)
        x = self.downsampling(x)
        if self.compression is not None:
            x = self.compression(x)
        return x

    
class ComputeSubbands(torch.nn.Module):
    """
    Convolves input with impulse response of filters in cochlear filter
    bank.
    """
    def __init__(self, coch_filters, binaural=False):
        super(ComputeSubbands, self).__init__()
        self.n_taps = coch_filters.shape[1]
        self.pad_factor = self.n_taps - 1 # need odd number for total len
        self.n_channels = coch_filters.shape[0]
        coch_filters = coch_filters.unsqueeze(1)
        self.register_buffer("coch_filters", coch_filters)

    def forward(self, x):
        x_shape = x.shape
        if len(x_shape)>2:
            x = x.view(x_shape[0]*x_shape[-2], 1, -1)
        else: # Handle the case where there is no batch dimension
            x = x.view(x_shape[0], 1, -1)
        # Pad time bins  of input 
        x = torch.nn.functional.pad(x, ((self.pad_factor,0)), mode='constant', value=0)
        x = torch.nn.functional.conv1d(x, self.coch_filters, padding='valid')
        x = x.view(x_shape[0], 1, self.n_channels, -1)
        return x
    
    
def ERB_filter_bank(x: torch.Tensor, fcoefs: torch.Tensor) -> torch.Tensor:
    r"""Processes a wavform with an ERB filter bank using an IIR filter using pytorch.
    
    Args:
        x (torch.Tensor): with shape `(Batch, Channels, Time)`.
        fcoefs (torch.Tensor): tensor of ERB filterbank coefficients
            with shape '(n_channels, 10)'.

    Returns:
        (torch.Tensor):
            torch.Tensor
                y, with shape `(Batch, Channels, Time)`.

    """

    if fcoefs.shape[1] != 10:
        raise ValueError('fcoefs parameter passed to ERBFilterBank is the wrong size.')
    
    a1 = fcoefs[:, [0,1,5]]
    a2 = fcoefs[:, [0,2,5]]
    a3 = fcoefs[:, [0,3,5]]
    a4 = fcoefs[:, [0,4,5]]

    b_coefs = fcoefs[:,[6,7,8]]
    
    gain = fcoefs[:,9].unsqueeze(1)

    # Apply cascade of low-pass filters 
    # for ref on clamp and batching args:
    # https://pytorch.org/audio/0.10.0/functional.html#lfilter 
    # Current settings match output from matlab and numpy implementations
    
    # Note: following Slaney implementation where b = A and a = B 
    # in lfilter input args 
    y = F.lfilter(x, b_coefs, 
                 torch.div(a1, gain), clamp=False, batching=False)

    y = F.lfilter(y, b_coefs, a2, clamp=False, batching=True)

    y = F.lfilter(y, b_coefs, a3, clamp=False, batching=True)

    y = F.lfilter(y, b_coefs, a4, clamp=False, batching=True)

    return y 

    
    
def ERB_space(lowFreq=100, highFreq=44100/4, N=100):
    '''
    Python port from Malcom Slaney's Audio Toolbox 
    Port by Ian Griffith. August 24, 2021. 

    This function computes an array of N frequencies uniformly spaced between
    highFreq and lowFreq on an ERB scale.  N is set to 100 if not specified.

    See also linspace, logspace, MakeERBCoeffs, MakeERBFilters.

    For a definition of ERB, see Moore, B. C. J., and Glasberg, B. R. (1983).
    "Suggested formulae for calculating auditory-filter bandwidths and
    excitation patterns," J. Acoust. Soc. Am. 74, 750-753.'''


    # Change the following three parameters if you wish to use a different
    # ERB scale.  Must change in MakeERBCoeffs too.
    EarQ = 9.26449            #  Glasberg and Moore Parameters
    minBW = 24.7
    order = 1
#     print(highFreq)
    # All of the followFreqing expressions are derived in Apple TR #35, "An
    # Efficient Implementation of the Patterson-Holdsworth Cochlear
    # Filter Bank."  See pages 33-34.
    freq_ixs = np.arange(N).reshape(1,-1) + 1 # +1 to match matlab (1:N) construction
    
    cfArray = -(EarQ*minBW) + np.exp(freq_ixs.T * (-np.log(highFreq + EarQ*minBW) + 
            np.log(lowFreq + EarQ*minBW))/ N) * (highFreq + EarQ*minBW)
    return cfArray



def make_ERB_filters(fs,numChannels,lowFreq):
    ''' 
     Python port from Malcom Slaney's Audio Toolbox 
     Port by Ian Griffith. August 24, 2021. 
     
     This function computes the filter coefficients for a bank of 
     Gammatone filters.  These filters were defined by Patterson and 
     Holdworth for simulating the cochlea.  

     The result is returned as an array of filter coefficients.  Each row 
     of the filter arrays contains the coefficients for four second order 
     filters.  The transfer function for these four filters share the same
     denominator (poles) but have different numerators (zeros).  All of these
     coefficients are assembled into one vector that the ERBFilterBank 
     can take apart to implement the filter.

     The filter bank contains "numChannels" channels that extend from
     half the sampling rate (fs) to "lowFreq".  Alternatively, if the numChannels
     input argument is a vector, then the values of this vector are taken to
     be the center frequency of each desired filter.  (The lowFreq argument is
     ignored in this case.)
    '''
    
    t = 1/fs
           
    # cf is array of center frequencies 
#     print(lowFreq)
    
    if isinstance(numChannels, np.ndarray):
        cf = numChannels[1:]
        if cf.shape[1] > cf.shape[0]:
            cf = cf.T
    else:
         cf = ERB_space(lowFreq, fs/2, numChannels) 
            
            
    # Change the followFreqing three parameters if you wish to use a different
    # ERB scale.  Must change in ERBSpace too.
    earQ = 9.26449        #  Glasberg and Moore Parameters
    minBW = 24.7
    order = 1
    
    erb = (((cf/earQ)**order) + minBW ** order) ** (1/order)
    b=1.019*2*np.pi*erb
    
    a0 = t
    a2 = 0
    b0 = 1
    b1 = -2*np.cos(2*cf*np.pi*t)/np.exp(b*t)
    b2 = np.exp(-2*b*t)
    
    
    a11 = -(2*t*np.cos(2*cf*np.pi*t)/np.exp(b*t) + 2*np.sqrt(3+2**1.5)*t*np.sin(2*cf*np.pi*t)/
            np.exp(b*t))/2
    a12 = -(2*t*np.cos(2*cf*np.pi*t)/np.exp(b*t) - 2*np.sqrt(3+2**1.5)*t*np.sin(2*cf*np.pi*t)/ 
            np.exp(b*t))/2
    a13 = -(2*t*np.cos(2*cf*np.pi*t)/np.exp(b*t) + 2*np.sqrt(3-2**1.5)*t*np.sin(2*cf*np.pi*t)/ 
            np.exp(b*t))/2
    a14 = -(2*t*np.cos(2*cf*np.pi*t)/np.exp(b*t) - 2*np.sqrt(3-2**1.5)*t*np.sin(2*cf*np.pi*t)/ 
            np.exp(b*t))/2

    gain = abs((-2*np.exp(4*1j*cf*np.pi*t)*t + 
                     2*np.exp(-(b*t) + 2*1j*cf*np.pi*t)*t* 
                             (np.cos(2*cf*np.pi*t) - np.sqrt(3 - 2**(3/2))* 
                              np.sin(2*cf*np.pi*t))) * 
               (-2*np.exp(4*1j*cf*np.pi*t)*t + 
                 2*np.exp(-(b*t) + 2*1j*cf*np.pi*t)*t* 
                  (np.cos(2*cf*np.pi*t) + np.sqrt(3 - 2**(3/2)) * 
                   np.sin(2*cf*np.pi*t)))* 
               (-2*np.exp(4*1j*cf*np.pi*t)*t + 
                 2*np.exp(-(b*t) + 2*1j*cf*np.pi*t)*t* 
                  (np.cos(2*cf*np.pi*t) - 
                   np.sqrt(3 + 2**(3/2))*np.sin(2*cf*np.pi*t))) * 
               (-2*np.exp(4*1j*cf*np.pi*t)*t + 2*np.exp(-(b*t) + 2*1j*cf*np.pi*t)*t* 
               (np.cos(2*cf*np.pi*t) + np.sqrt(3 + 2**(3/2))*np.sin(2*cf*np.pi*t))) / 
              (-2 / np.exp(2*b*t) - 2*np.exp(4*1j*cf*np.pi*t) +  
               2*(1 + np.exp(4*1j*cf*np.pi*t))/np.exp(b*t))**4)
    allfilts = np.ones((max(cf.shape),1)) 
    fcoefs = np.hstack([a0*allfilts, a11, a12, a13, a14, a2*allfilts, b0*allfilts, b1, b2, gain])
    return fcoefs , cf



def backend_hann2d(x, ramp_dur_ms, samplerate=48000):
    stim_dur_smp = x.shape[1] # N taps of x
    ramp_dur_smp =  np.floor(ramp_dur_ms * samplerate / 1000).astype('int')
    assert stim_dur_smp > (2*ramp_dur_smp), 'Ramps cannot be longer than the stimulus duration'
    
    # calc window
    # https://stackoverflow.com/questions/56485663/hanning-window-values-doesnt-match-in-python-and-matlab
    win = sig.hann((2 * ramp_dur_smp) + 2)[1:-1]
    
    # Middle part (steady state)
    steady_win = x[:,:stim_dur_smp-ramp_dur_smp]
    
    # Final part of windowed stimulus
    end_win = win[ramp_dur_smp : ramp_dur_smp*2] * x[:, stim_dur_smp-ramp_dur_smp:stim_dur_smp]

    return np.hstack([steady_win, end_win])
