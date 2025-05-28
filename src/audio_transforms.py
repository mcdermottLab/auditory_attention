import torch
import torchaudio
import random
import numpy as np
import sys
import scipy.signal as sps
sys.path.append('/om/user/imgriff/python-packages/chcochleagram')
import chcochleagram
from chcochleagram import compression
from chcochleagram import cochleagram
from chcochleagram import *
from .time_domain_cochleagram import TimeDomainCochleagram
import torchaudio.transforms as T

# sys.path.append('/om2/user/annesyab/summer2022/CI_Model_HearingImpairment/cochlear_implant_param_infer/utils')


def ch_demean(x, dim=0, mean_keepdim=False):
    '''
    Helper function to mean-subtract tensor.
    Args
    ----
    x (tensor): tensor to be mean-subtracted
    dim (int): kwarg for torch.mean (dim along which to compute mean)
    Returns
    -------
    x_demean (tensor): mean-subtracted tensor
    '''
    x_demean = torch.sub(x, torch.mean(x, dim=dim, keepdim=mean_keepdim))
    return x_demean


def ch_global_demean(x, v2=False, dim=(-2,-1)):
    '''
    Helper function to globally mean-subtract tensor.

    Args
    ----
    x (tensor): tensor to be mean-subtracted
    v2 (bool):  whether to use updated de-mean method
    dim (tuple): kwarg for torch.mean (dim along which to compute mean)

    Returns
    -------
    x_demean (tensor): mean-subtracted tensor
    '''
    if v2:
        return ch_demean(x, dim=dim, mean_keepdim=True)
    x_demean = torch.sub(x, torch.mean(x))
    return x_demean


def ch_rms(x, dim=0):
    '''
    Helper function to compute RMS amplitude of a tensor.
    Args
    ----
    x (tensor): tensor for which RMS amplitude should be computed
    dim (int): kwarg for torch.mean (dim along which to compute mean)
    Returns
    -------
    rms_x (tensor): root-mean-square amplitude of x
    '''
    rms_x = torch.sqrt(torch.mean(torch.pow(x, 2), dim=dim))
    return rms_x


def ch_global_rms(x):
    '''
    Helper function to compute RMS amplitude of a tensor.
    Args
    ----
    x (tensor): tensor for which RMS amplitude should be computed
    dim (int): kwarg for torch.mean (dim along which to compute mean)
    Returns
    -------
    rms_x (tensor): root-mean-square amplitude of x
    '''
    rms_x = torch.sqrt(torch.mean(torch.pow(x, 2)))
    return rms_x


class AudioCompose(torch.nn.Module):
    """
    Composes several foreground/background audio transforms together (based off of
        torchvision.transforms.Compose)
    Args:
        transforms (list of audio_function transfrom torch.nn.Modules): list of transforms to compose.
    """

    def __init__(self, transforms):
        super(AudioCompose, self).__init__()
        self.transforms = transforms

    def forward(self, foreground_wav, background_wav):
        for t in self.transforms:
            foreground_wav, background_wav = t(foreground_wav, background_wav)
        return foreground_wav, background_wav

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class LogScaleFakeClipping(torch.nn.Module):
    """
    Scales the values by a log scale. (Useful to apply aftr the Mel Spectrogram)
    """
    def __init__(self, offset=1e-6):
        super(LogScaleFakeClipping, self).__init__()
        self.offset = offset
        self.clamp_function = FakeClamp.apply

    def forward(self, foreground_wav, background_wav):
        foreground_wav = self.clamp_function(foreground_wav, self.offset)
        foreground_wav = torch.log2(foreground_wav)
        if background_wav is not None:
            background_wav = self.clamp_function(background_wav, self.offset)
            background_wav = torch.log2(background_wav)
        return foreground_wav, background_wav

class FakeClamp(torch.autograd.Function):
    """
    Applies clamp in the forward pass, but all gradients=1 in the backwards
    pass.
    """
    @staticmethod
    def forward(ctx, x, min):
        return torch.clamp(x, min=min)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None

class LogScale(torch.nn.Module):
    """
    Scales the values by a log scale. (Useful to apply aftr the Mel Spectrogram)
    """
    def __init__(self, offset=1e-6):
        super(LogScale, self).__init__()
        self.offset = offset

    def forward(self, foreground_wav, background_wav):
        foreground_wav = torch.clamp(foreground_wav, min=self.offset)
        foreground_wav = torch.log2(foreground_wav)
        if background_wav is not None:
            background_wav = torch.clamp(background_wav, min=self.offset)
            background_wav = torch.log2(background_wav)
        return foreground_wav, background_wav

class ClippedGradPower(torch.nn.Module):
    """
    Wrapper around ClippedGradPowerCompression defined in chcochleagram.compression
    """
    def __init__(self, compression_kwargs):
        super(ClippedGradPower, self).__init__()
        self.compression_kwargs = compression_kwargs
        self.compression_function = compression.ClippedGradPowerCompression(**compression_kwargs)

    def forward(self, foreground_wav, background_wav):
        foreground_wav = self.compression_function(foreground_wav)
        if background_wav is not None:
            background_wav = self.compression_function(background_wav)
        return foreground_wav, background_wav


class AudioToAudioRepresentation(torch.nn.Module):
    """
    Base class for audio transformations. Takes in the audio and outputs
    a representation that is used for training.
    Args:
        rep_type (str): the type of representation to build
    """
    def __init__(self, rep_type, rep_kwargs, compression_type, compression_kwargs):
        super(AudioToAudioRepresentation, self).__init__()
        self.rep_type = rep_type
        self.rep_kwargs = rep_kwargs
        self.compression_type = compression_type
        self.compression_kwargs = compression_kwargs

        # Choose the representation type
        if self.rep_type == 'mel_spec':
            self.rep = AudioToMelSpectrogram(melspec_kwargs=self.rep_kwargs)
        elif self.rep_type == 'cochleagram':
            self.rep = AudioToCochleagram(cgram_kwargs=self.rep_kwargs)
        elif self.rep_type == 'cochlea_filt':
            self.rep = AudioToCochlearRep(cgram_kwargs=self.rep_kwargs)
        elif self.rep_type == 'cochlear_implant':
            self.rep = AudioToCIRep(cgram_kwargs=self.rep_kwargs)
        else:
            raise NotImplementedError('Audio Representation of type '
              '%s is not implemented'%self.rep_type)

        # Choose the compression type
        if self.compression_type == 'log':
            self.compression = LogScale(**self.compression_kwargs)
        elif self.compression_type == 'log_fakeclamp':
            self.compression = LogScaleFakeClipping(**self.compression_kwargs)
        elif self.compression_type == 'coch_p3':
            self.compression = ClippedGradPower(self.compression_kwargs)
        elif self.compression_type == 'none':
            self.compression = None
        else:
            raise NotImplementedError('Audio Compression of type '
               '%s is not implemented'%self.compression_type)

    def forward(self, foreground_wav, background_wav):
        if foreground_wav is not None: # For compat with FilterNanMusic
            foreground_rep, background_rep = self.rep(foreground_wav, background_wav)
            if self.compression is not None:
                foreground_rep, background_rep = self.compression(foreground_rep, background_rep)
            return foreground_rep, background_rep
        else:
            return None, None


class AudioToMelSpectrogram(torch.nn.Module):
    """
    Converts audio to mel spectrogram.
    Args:
        melspec_kwargs (dict): dictionary containing the arguments used within
            torchaudio.MelSpectrogram
    """
    def __init__(self, melspec_kwargs={}):
        super(AudioToMelSpectrogram, self).__init__()
        self.melspec_kwargs = melspec_kwargs
        self.MelSpectrogram = torchaudio.transforms.MelSpectrogram(**self.melspec_kwargs)

    def forward(self, foreground_wav, background_wav):
        """
        Args:
            foreground_wav (torch.Tensor): the waveform that will be used as
                the foreground audio sample (usually speech)
            background_wav (torch.Tensor): the waveform that will be used as
                the background audio sample
        """
        del background_wav

        foreground_mel = self.MelSpectrogram(foreground_wav)

        return foreground_mel, None


class AudioToCochleagram(torch.nn.Module):
    """
    Converts audio to cochleagram
    """
    def __init__(self, cgram_kwargs={}):
        super(AudioToCochleagram, self).__init__()
        self.cgram_kwargs = cgram_kwargs

        # Args used for multiple of the cochleagram operations
        self.signal_size = self.cgram_kwargs['signal_size']
        self.sr = self.cgram_kwargs['sr']
        self.pad_factor = self.cgram_kwargs['pad_factor']
        self.use_rfft = self.cgram_kwargs['use_rfft']

        # Define cochlear filters
        self.coch_filter_kwargs = self.cgram_kwargs['coch_filter_kwargs']
        self.coch_filter_kwargs = {'use_rfft':self.use_rfft,
                                   'pad_factor':self.pad_factor,
                                   'filter_kwargs':self.coch_filter_kwargs}

        self.make_coch_filters = self.cgram_kwargs['coch_filter_type']
        self.filters = self.make_coch_filters(self.signal_size,
                                              self.sr,
                                              **self.coch_filter_kwargs)

        # Define an envelope extraction operation
        self.env_extraction = self.cgram_kwargs['env_extraction_type']
        self.envelope_extraction = self.env_extraction(self.signal_size,
                                                       self.sr,
                                                       self.use_rfft,
                                                       self.pad_factor)

        # Define a downsampling operation
        self.downsampling = self.cgram_kwargs['downsampling_type']
        self.env_sr = self.cgram_kwargs['env_sr']
        self.downsampling_kwargs = self.cgram_kwargs['downsampling_kwargs']
        self.downsampling_op = self.downsampling(self.sr, self.env_sr, **self.downsampling_kwargs)

        # Compression is applied as a separate transform to be consistent with Spectrograms
        cochleagram = chcochleagram.cochleagram.Cochleagram(self.filters,
                                                            self.envelope_extraction,
                                                            self.downsampling_op,
                                                            compression=None)

        self.Cochleagram = cochleagram

    def forward(self, foreground_wav, background_wav):
        """
        Args:
            foreground_wav (torch.Tensor): the waveform that will be used as
                the foreground audio sample (usually speech)
            background_wav (torch.Tensor): the waveform that will be used as
                the background audio sample
        """
        del background_wav

        foreground_mel = self.Cochleagram(foreground_wav)

        return foreground_mel, None


class AudioToCochlearRep(torch.nn.Module):
    """
    Converts audio to simulated cochlear represenation. Convolves input with
    Gammatone filter bank, half-wave rectifies filter bank output, and applies
    downsampling.
    """
    def __init__(self, cgram_kwargs={}):
        super(AudioToCochlearRep, self).__init__()
        self.cgram_kwargs = cgram_kwargs

        # Args used for multiple of the cochleagram operations
        self.sr = self.cgram_kwargs['sr']
        self.env_sr = self.cgram_kwargs['env_sr']
        self.use_pad = self.cgram_kwargs['use_pad']
        # Define cochlear filters
        self.coch_filter_kwargs = {'sr':self.sr,
                                   'env_sr': self.env_sr,
                                   'n_channels': cgram_kwargs['n_channels'],
                                   'low_lim': cgram_kwargs['low_lim'],
                                   }
        # Define an envelope extraction operation in Forward
        # Define a downsampling operation
        if isinstance(self.cgram_kwargs['downsampling_type'], str):
            self.downsampling = downsampling_reps[self.cgram_kwargs['downsampling_type']]
        else:
            self.downsampling = self.cgram_kwargs['downsampling_type']
        self.downsampling_kwargs = self.cgram_kwargs['downsampling_kwargs']
        self.downsampling_op = self.downsampling(self.sr,
                                                 self.env_sr,
                                                 **self.downsampling_kwargs,
                                                 )
        # Compression is applied as a separate transform to be consistent with Spectrograms
        # Define cochleagram
        self.Cochleagram = TimeDomainCochleagram(self.coch_filter_kwargs,
                                                self.downsampling_op,
                                                compression=None,
                                                **cgram_kwargs)


    def forward(self, foreground_wav, background_wav):
        """
        Args:
            foreground_wav (torch.Tensor): the waveform that will be used as
                the foreground audio sample (usually speech)
            background_wav (torch.Tensor): the waveform that will be used as
                the background audio sample
        """
        del background_wav
        
        foreground_coch = self.Cochleagram(foreground_wav)

        return foreground_coch, None


class AudioToCIRep(torch.nn.Module):
    """
    Converts audio to cochlear implant nervegram
    """
    def __init__(self, cgram_kwargs={}):
        super(AudioToCIRep, self).__init__()
        self.cgram_kwargs = cgram_kwargs

        ci_simulator = ci_model.SequentialCIModel(self.cgram_kwargs)

        self.CISimulator = ci_simulator

    def forward(self, foreground_wav, background_wav):
        """
        Args:
            foreground_wav (torch.Tensor): the waveform that will be used as
                the foreground audio sample (usually speech)
            background_wav (torch.Tensor): the waveform that will be used as
                the background audio sample
        """
        del background_wav

        # hanlde the case where the input is binaural 
        if foreground_wav.ndim >= 2 and foreground_wav.shape[-2] == 2:
            fg_left = self.CISimulator(foreground_wav[:,0,:].squeeze())
            fg_rigth = self.CISimulator(foreground_wav[:,1,:].squeeze())
            foreground_ci_nervegram = torch.concat([fg_left, fg_rigth], dim=1)
        else:
            foreground_ci_nervegram = self.CISimulator(foreground_wav.sqeeze())

        return foreground_ci_nervegram, None
    

class AudioToTensor(torch.nn.Module):
    """
    Convert the foreground and background sounds to tensors
    Args:
        None
    Returns:
        foreground_wav, background_wav
    """
    def __init__(self):
        super(AudioToTensor, self).__init__()

    def forward(self, foreground_wav, background_wav):
        """
        Args:
            foreground_wav (torch.Tensor): the waveform that will be used as
                the foreground audio sample (usually speech)
            background_wav (torch.Tensor): the waveform that will be used as
                the background audio sample
        """
        # check if foregoround_wav is a torch tensor
        if not isinstance(foreground_wav, torch.Tensor):
            foreground_wav = torch.from_numpy(foreground_wav)
        if background_wav is None:
            return foreground_wav, None
        else:
            if not isinstance(background_wav, torch.Tensor):
                background_wav = torch.from_numpy(background_wav)
            return foreground_wav, background_wav


class UnsqueezeAudio(torch.nn.Module):
    """
    Adds a channel dimension (useful for mel-spectrograms as inputs)
    Args:
        None
    Returns:
        foreground_wav, background_wav
    """
    def __init__(self, dim=1):
        super(UnsqueezeAudio, self).__init__()
        self.dim = dim

    def forward(self, foreground_wav, background_wav):
        if foreground_wav is not None:
            foreground_wav = foreground_wav.unsqueeze(self.dim)
        if background_wav is not None:
            background_wav = background_wav.unsqueeze(self.dim)
        return foreground_wav, background_wav


class FilterNoneSpeech(torch.nn.Module):
    """
    Filter out speech audio samples that are all zeros.
    Useful for removing speech 'null' classes.
    Args:
        None
    Returns:
        foreground_wav, background_wav if passes filtering
        None if should be removed
    """
    def __init__(self):
        super(FilterNoneSpeech, self).__init__()

    def forward(self, foreground_wav, background_wav):
        if torch.sum(torch.pow(foreground_wav, 2))==0:
            foreground_wav = None
        if torch.sum(torch.pow(background_wav, 2))==0:
            background_wav = None
        else:
            return foreground_wav, background_wav


class FilterNanMusic(torch.nn.Module):
    """
    Filter out music audio samples that are nans.
    Useful for removing music 'null' classes.
    Args:
        None
    Returns:
        foreground_wav, background_wav if passes filtering
        None if should be removed
    """
    def __init__(self):
        super(FilterNanMusic, self).__init__()

    def __call__(self, foreground_wav, background_wav):
        background_wav = None # Always none for music
        if torch.isnan(foreground_wav).any():
            foreground_wav = None

        return foreground_wav, background_wav


class RandomCropForegroundBackground(torch.nn.Module):
    """
    Randomly crops the foreground and background to make a shorter signal.
    """
    def __init__(self, signal_size, crop_length):
        super(RandomCropForegroundBackground, self).__init__()
        self.crop_length = crop_length
        self.signal_size = signal_size
        self.start_crop = int(signal_size - crop_length)

    def forward(self, foreground_wav, background_wav):
        """
        Args:
            foreground_wav (torch.Tensor): the waveform that will be used as
                the foreground audio sample (usually speech)
            background_wav (torch.Tensor): the waveform that will be used as
                the background audio sample
        """
        rand_start = torch.randint(self.start_crop, size=(2,))
        foreground_wav = foreground_wav[rand_start[0]:rand_start[0]+self.crop_length]
        background_wav = background_wav[rand_start[1]:rand_start[1]+self.crop_length]
        return foreground_wav, background_wav

class CenterCropForegroundRandomCropBackground(torch.nn.Module):
    """
    Center crops the foreground and randomly crops background to make a shorter signal.
    """
    def __init__(self, signal_size, crop_length):
        super(CenterCropForegroundRandomCropBackground, self).__init__()
        self.crop_length = crop_length
        self.signal_size = signal_size
        self.start_crop_random = int(signal_size - crop_length)
        self.start_crop_center = int((signal_size-crop_length)/2)

    def forward(self, foreground_wav, background_wav):
        """
        Args:
            foreground_wav (torch.Tensor): the waveform that will be used as
                the foreground audio sample (usually speech)
            background_wav (torch.Tensor): the waveform that will be used as
                the background audio sample
        """
        rand_start = torch.randint(self.start_crop_random, size=(2,))
        foreground_wav = foreground_wav[self.start_crop_center:self.start_crop_center+self.crop_length]
        background_wav = background_wav[rand_start[1]:rand_start[1]+self.crop_length]
        return foreground_wav, background_wav


class TimeReverseBackgroundSignal(torch.nn.Module):
    """
    Time reverses the background signal.
    """
    def __init__(self, time_dim=[-1]):
        super(TimeReverseBackgroundSignal, self).__init__()
        self.time_dim = time_dim

    def forward(self, foreground_wav, background_wav):
        """
        Args:
            foreground_wav (torch.Tensor): the waveform that will be used as
                the foreground audio sample (usually speech)
            background_wav (torch.Tensor): the waveform that will be used as
                the background audio sample
        """
        if background_wav is not None:
            background_wav = torch.flip(background_wav, self.time_dim)
        return foreground_wav, background_wav


class RMSNormalizeForegroundAndBackground(torch.nn.Module):
    """
    RMS normalize the foreground and background sounds
    Args:
        rms_normalization (float): The rms level to set the sound to
    Returns:
        foreground_wav, background_wav
    """
    def __init__(self, rms_level=0.1):
        super(RMSNormalizeForegroundAndBackground, self).__init__()
        self.rms_level=rms_level

    def forward(self, foreground_wav, background_wav):
        """
        Args:
            foreground_wav (torch.Tensor): the waveform that will be used as
                the foreground audio sample (usually speech)
            background_wav (torch.Tensor): the waveform that will be used as
                the background audio sample
        """
        if foreground_wav is not None:
            foreground_wav = ch_demean(foreground_wav)
            rms_foreground = ch_rms(foreground_wav)
            if rms_foreground !=0:
                foreground_wav = foreground_wav * self.rms_level / rms_foreground
            else:
                foreground_wav = foreground_wav # for music genre, keep dead samples
#                 raise ValueError("Trying to RMS Normalize a signal that is all zeros")

        if background_wav is not None:
            background_wav = ch_demean(background_wav)
            rms_background = ch_rms(background_wav)
            if rms_background !=0:
                background_wav = background_wav * self.rms_level / rms_background
            else:
                background_wav = None
#                 raise ValueError("Trying to RMS Normalize a signal that is all zeros")

        return foreground_wav, background_wav


class DBSPLNormalizeForegroundAndBackground(torch.nn.Module):
    """
    Set the foreground and background sounds to a specified sound pressure 
    level (dBSPL)

    Args:
        dbspl (float): desired sound pressure level in dB re 20e-6 Pa

    Returns:
        foreground_wav, background_wav
    """
    def __init__(self, dbspl=60):
        super(DBSPLNormalizeForegroundAndBackground, self).__init__()
        self.dbspl=dbspl
        self.rms_level = 20e-6 * np.power(10.0, self.dbspl / 20.0)

    def forward(self, foreground_wav, background_wav):
        """
        Args:
            foreground_wav (torch.Tensor): the waveform that will be used as
                the foreground audio sample (usually speech)
            background_wav (torch.Tensor): the waveform that will be used as
                the background audio sample
        """
        if foreground_wav is not None:
            if foreground_wav.ndim == 3 and foreground_wav.shape[-2] == 2:
                foreground_wav = ch_global_demean(foreground_wav, v2=True)
                rms_foreground = ch_global_rms(foreground_wav)
            else:
                foreground_wav = ch_demean(foreground_wav)
                rms_foreground = ch_rms(foreground_wav)

            if rms_foreground !=0:
                foreground_wav = foreground_wav * self.rms_level / rms_foreground
            else:
                foreground_wav = None

        if background_wav is not None:
            if background_wav.ndim == 3 and background_wav.shape[-2] == 2:
                background_wav = ch_global_demean(background_wav, v2=True)
                rms_background = ch_global_rms(background_wav)
            else:
                background_wav = ch_demean(background_wav)
                rms_background = ch_rms(background_wav)
            if rms_background !=0:
                background_wav = background_wav * self.rms_level / rms_background
            else:
                background_wav = None

        return foreground_wav, background_wav


class BinauralRMSNormalizeForegroundAndBackground(torch.nn.Module):
    """
    RMS normalize the foreground and background sounds

    Args:
        rms_normalization (float): The rms level to set the sound to

    Returns:
        foreground_wav, background_wav
    """
    def __init__(self, rms_level=0.1, v2_demean=True):
        super(BinauralRMSNormalizeForegroundAndBackground, self).__init__()
        self.rms_level=rms_level
        self.v2_demean=v2_demean

    def forward(self, foreground_wav, background_wav):
        """
        Args:
            foreground_wav (torch.Tensor): the waveform that will be used as
                the foreground audio sample (usually speech)
            background_wav (torch.Tensor): the waveform that will be used as
                the background audio sample
        """
        if foreground_wav is not None:
            foreground_wav = ch_global_demean(foreground_wav, v2=self.v2_demean)
            rms_foreground = ch_global_rms(foreground_wav)
            if rms_foreground !=0:
                foreground_wav = foreground_wav * self.rms_level / rms_foreground
            else:
                foreground_wav = foreground_wav # for music genre, keep dead samples
#                 raise ValueError("Trying to RMS Normalize a signal that is all zeros")

        if background_wav is not None:
            background_wav = ch_global_demean(background_wav, v2=self.v2_demean)
            rms_background = ch_global_rms(background_wav)
            if rms_background !=0:
                background_wav = background_wav * self.rms_level / rms_background
            else:
                background_wav = None
#                 raise ValueError("Trying to RMS Normalize a signal that is all zeros")

        return foreground_wav, background_wav


class CombineWithRandomDBSNR(torch.nn.Module):
    """
    Takes two signals and combines them at a specified dB SNR level.
    Args:
        low_snr (float): the low end for the range of dB SNR to draw from
        high_snr (float): the high end for the range of db SNR to draw from
        rms_level (float): the end RMS value for the combined sound
    Returns:
        signal_in_noise, None
    """
    def __init__(self, low_snr=-10, high_snr=10):
        self.low_snr=low_snr
        self.high_snr=high_snr
        super(CombineWithRandomDBSNR, self).__init__()

    def forward(self, foreground_wav, background_wav):
        """
        Args:
            foreground_wav (torch.Tensor): the waveform that will be used as
                the foreground audio sample (usually speech)
            background_wav (torch.Tensor): the waveform that will be used as
                the background audio sample
        """
        if self.low_snr == "clean" or self.high_snr == "clean":
            return foreground_wav, None
        if background_wav is None:
            return foreground_wav, None
        rand_db_snr = random.uniform(self.low_snr, self.high_snr)
        rms_ratio = np.power(10.0, rand_db_snr / 20.0)
        # Demean signal and noise before computing rms
        foreground_wav = ch_demean(foreground_wav)
        background_wav = ch_demean(background_wav)

        rms_foreground = ch_rms(foreground_wav)
        rms_background = ch_rms(background_wav)

        # Calculate the scale factor for the two sounds
        # TODO: filter out the signals that are only foreground or only background.
        # For now, to align with the jsinv3 dataset, we include the infinite SNR
        # cases

        if rms_foreground == 0: # No foreground condition (just noise)
            noise_scale_factor = 1
        elif rms_background == 0:
            noise_scale_factor = 0
        else:
            noise_scale_factor = torch.div(rms_foreground,
                                           torch.mul(rms_background,
                                                     rms_ratio))

        background_wav = torch.mul(noise_scale_factor, background_wav)
        signal_in_noise = torch.add(foreground_wav, background_wav)

        return signal_in_noise, None
    

class BinauralCombineWithRandomDBSNR(torch.nn.Module):
    """
    Takes two signals and combines them at a specified dB SNR level.
    Args:
        low_snr (float): the low end for the range of dB SNR to draw from
        high_snr (float): the high end for the range of db SNR to draw from
        rms_level (float): the end RMS value for the combined sound

    Returns:
        signal_in_noise, None

    """
    def __init__(self, low_snr=-10, high_snr=10, v2_demean=False):
        self.low_snr=low_snr
        self.high_snr=high_snr
        self.v2_demean = v2_demean
        super(BinauralCombineWithRandomDBSNR, self).__init__()

    def forward(self, foreground_wav, background_wav):
        """
        Args:
            foreground_wav (torch.Tensor): the waveform that will be used as
                the foreground audio sample (usually speech)
            background_wav (torch.Tensor): the waveform that will be used as
                the background audio sample
        """
        if self.low_snr == "clean" or self.high_snr == "clean":
            return foreground_wav, None
        if background_wav is None:
            return foreground_wav, None
        rand_db_snr = random.uniform(self.low_snr, self.high_snr)
        rms_ratio = np.power(10.0, rand_db_snr / 20.0)
        # Demean signal and noise before computing rms
        foreground_wav = ch_global_demean(foreground_wav, v2=self.v2_demean)
        background_wav = ch_global_demean(background_wav, v2=self.v2_demean)

        rms_foreground = ch_global_rms(foreground_wav)
        rms_background = ch_global_rms(background_wav)

        # Calculate the scale factor for the two sounds
        # TODO: filter out the signals that are only foreground or only background.
        # For now, to align with the jsinv3 dataset, we include the infinite SNR
        # cases
        if rms_foreground == 0: # No foreground condition (just noise)
            noise_scale_factor = 1
        elif rms_background == 0:
            noise_scale_factor = 0
        else:
            noise_scale_factor = torch.div(rms_foreground,
                                           torch.mul(rms_background,
                                                     rms_ratio))

        background_wav = torch.mul(noise_scale_factor, background_wav)
        signal_in_noise = torch.add(foreground_wav, background_wav)

        return signal_in_noise, None


class DuplicateChannel(torch.nn.Module):
    """
    Duplicates the input channel to the number of output channels.

    Args:
        num_output_channels (int): the number of output channels
        dim (int): the axis to duplicate along

    Returns:
        foreground_wav, background_wav
    """

    def __init__(self, num_output_channels=2, dim=1):
        super(DuplicateChannel, self).__init__()
        self.num_output_channels = num_output_channels
        self.dim = dim

    def forward(self, foreground_wav, background_wav):
        if foreground_wav is not None:
            foreground_wav = foreground_wav.repeat(self.num_output_channels , 1)
        if background_wav is not None:
            background_wav = background_wav.repeat(self.num_output_channels , 1)
        return foreground_wav, background_wav
        

class Spatialize(torch.nn.Module):
    """
    Torch nn.Module for spatializing audio via convolution with a BRIR.

    Args:
        ir (numpy.ndarray): The impulse response used for spatialization.
        model_sr (int, optional): The sample rate of the model. Defaults to 50,000.
        start_crop_in_s (float, optional): The start time of the crop in seconds. Defaults to 0.25.
        end_crop_in_s (float, optional): The end time of the crop in seconds. Defaults to 2.75.

    Returns:
        spatialized (torch.Tensor): The spatialized audio. 
    """
    def __init__(self, ir, model_sr=50_000, start_crop_in_s=0.25, end_crop_in_s=2.75):
        super(Spatialize, self).__init__()
        ir = torch.flip(torch.from_numpy(ir), dims=[0]).float()
        self.n_taps = ir.shape[0]
        ir = ir.T.unsqueeze(1)
        # get center crop of 2.5 seconds relative to model_sr
        if start_crop_in_s and end_crop_in_s :
            self.start_frame = int(model_sr * start_crop_in_s)
            self.end_frame = int(model_sr * end_crop_in_s)
        else:
            self.start_frame = 0
            self.end_frame = None # crop to end of signal
        self.register_buffer("ir", ir)

    def forward(self, x):
        n_x = x.shape[0]
        # pad last dim of x with ir.shape[0] - 1 zeros
        x_padded = torch.nn.functional.pad(x, (self.n_taps - 1, 0))
        spatialized = torch.nn.functional.conv1d(x_padded.view(n_x, 1, -1), self.ir)
        # resize to desired shape
        spatialized = spatialized[:, :, self.start_frame:self.end_frame]
        return spatialized
    
class Resample(torch.nn.Module):
    """
    Torch nn.Module for spatializing audio via convolution with a BRIR.

    Args:
        ir (numpy.ndarray): The impulse response used for spatialization.
        model_sr (int, optional): The sample rate of the model. Defaults to 50,000.
        start_crop_in_s (float, optional): The start time of the crop in seconds. Defaults to 0.25.
        end_crop_in_s (float, optional): The end time of the crop in seconds. Defaults to 2.75.

    Returns:
        spatialized (torch.Tensor): The spatialized audio. 
    """
    def __init__(self, orig_freq=50_000, new_freq=10_000, **kwargss):
        super(Resample, self).__init__()
        self.orig_freq = orig_freq
        self.new_freq = new_freq
        self.resample = T.Resample(self.orig_freq, self.new_freq, **kwargss)

    def forward(self, foreground_wav, background_wav):
        if foreground_wav is not None:
            foreground_wav =  self.resample(foreground_wav.contiguous())
        if background_wav is not None:
            background_wav =  self.resample(background_wav.contiguous())
        return foreground_wav, background_wav


# Add dict of downsampling operations to be performed on audio representations
downsampling_reps = {'SincWithKaiserWindow': chcochleagram.downsampling.SincWithKaiserWindow, 
                     'TorchTransformsResample': T.Resample}

