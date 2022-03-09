import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T


def ch_demean(x, dim=0):
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
    x_demean = torch.sub(x, torch.mean(x, dim=dim))
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
        rand_db_snr = random.uniform(self.low_snr, self.high_snr)
        rms_ratio = torch.pow(10.0, rand_db_snr / 20.0)
        # Demean signal and noise before computing rms
        foreground_wav = torch.sub(foreground_wav, torch.mean(foreground_wav, dim=0))
        background_wav = torch.sub(background_wav, torch.mean(background_wav, dim=0))

        rms_foreground = torch.sqrt(torch.mean(torch.pow(foreground_wav, 2), dim=0))
        rms_background = torch.sqrt(torch.mean(torch.pow(background_wav, 2), dim=0))

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

        return signal_in_noise


class CMVN(torch.jit.ScriptModule):

    __constants__ = ["mode", "dim", "eps"]

    def __init__(self, mode="global", dim=2, eps=1e-10):
        # `torchaudio.load()` loads audio with shape [channel, feature_dim, time]
        # so perform normalization on dim=2 by default
        super(CMVN, self).__init__()

        if mode != "global":
            raise NotImplementedError(
                "Only support global mean variance normalization.")

        self.mode = mode
        self.dim = dim
        self.eps = eps

    @torch.jit.script_method
    def forward(self, x):
        if self.mode == "global":
            return (x - x.mean(self.dim, keepdim=True)) / (self.eps + x.std(self.dim, keepdim=True))

    def extra_repr(self):
        return "mode={}, dim={}, eps={}".format(self.mode, self.dim, self.eps)


class Delta(torch.jit.ScriptModule):

    __constants__ = ["order", "window_size", "padding"]

    def __init__(self, order=1, window_size=2):
        # Reference:
        # https://kaldi-asr.org/doc/feature-functions_8cc_source.html
        # https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/layers/common_audio.py
        super(Delta, self).__init__()

        self.order = order
        self.window_size = window_size

        filters = self._create_filters(order, window_size)
        self.register_buffer("filters", filters)
        self.padding = (0, (filters.shape[-1] - 1) // 2)

    @torch.jit.script_method
    def forward(self, x):
        # Unsqueeze batch dim
        x = x.unsqueeze(0)
        return F.conv2d(x, weight=self.filters, padding=self.padding)[0]

    # TODO(WindQAQ): find more elegant way to create `scales`
    def _create_filters(self, order, window_size):
        scales = [[1.0]]
        for i in range(1, order + 1):
            prev_offset = (len(scales[i-1]) - 1) // 2
            curr_offset = prev_offset + window_size

            curr = [0] * (len(scales[i-1]) + 2 * window_size)
            normalizer = 0.0
            for j in range(-window_size, window_size + 1):
                normalizer += j * j
                for k in range(-prev_offset, prev_offset + 1):
                    curr[j+k+curr_offset] += (j * scales[i-1][k+prev_offset])
            curr = [x / normalizer for x in curr]
            scales.append(curr)

        max_len = len(scales[-1])
        for i, scale in enumerate(scales[:-1]):
            padding = (max_len - len(scale)) // 2
            scales[i] = [0] * padding + scale + [0] * padding

        return torch.tensor(scales).unsqueeze(1).unsqueeze(1)

    def extra_repr(self):
        return "order={}, window_size={}".format(self.order, self.window_size)


class Postprocess(torch.jit.ScriptModule):
    @torch.jit.script_method
    def forward(self, x):
        # [channel, feature_dim, time] -> [time, channel, feature_dim]
        x = x.permute(2, 0, 1)
        # [time, channel, feature_dim] -> [time, feature_dim * channel]
        return x.reshape(x.size(0), -1).detach()


# TODO(Windqaq): make this scriptable
class ExtractMelFromWav(nn.Module):
    def __init__(self, mode="fbank", num_mel_bins=40, sample_rate=16000, **kwargs):
        super(ExtractMelFromWav, self).__init__()
        self.mode = mode
        self.extract_fn = torchaudio.compliance.kaldi.fbank if mode == "fbank" else torchaudio.compliance.kaldi.mfcc
        self.num_mel_bins = num_mel_bins
        self.kwargs = kwargs
        self.sample_rate = sample_rate

    def forward(self, wav_tensor):
        y = self.extract_fn(wav_tensor,
                            num_mel_bins=self.num_mel_bins,
                            channel=-1,
                            sample_frequency=self.sample_rate,
                            **self.kwargs)
        return y.transpose(0, 1).unsqueeze(0).detach()

    def extra_repr(self):
        return "mode={}, num_mel_bins={}".format(self.mode, self.num_mel_bins)


# TODO(Windqaq): make this scriptable
class ExtractAudioFeature(nn.Module):
    def __init__(self, mode="fbank", num_mel_bins=40, **kwargs):
        super(ExtractAudioFeature, self).__init__()
        self.mode = mode
        self.extract_fn = torchaudio.compliance.kaldi.fbank if mode == "fbank" else torchaudio.compliance.kaldi.mfcc
        self.num_mel_bins = num_mel_bins
        self.kwargs = kwargs

    def forward(self, filepath):
        waveform, sample_rate = torchaudio.load(filepath)

        y = self.extract_fn(waveform,
                            num_mel_bins=self.num_mel_bins,
                            channel=-1,
                            sample_frequency=sample_rate,
                            **self.kwargs)
        return y.transpose(0, 1).unsqueeze(0).detach()

    def extra_repr(self):
        return "mode={}, num_mel_bins={}".format(self.mode, self.num_mel_bins)

    
# TODO(Windqaq): make this scriptable
class ExtractAndResampleFeature(nn.Module):
    def __init__(self, mode="fbank", num_mel_bins=40, saved_rate=44100, new_rate=16000, **kwargs):
        super(ExtractAndResampleFeature, self).__init__()
        self.mode = mode
        # Using default params from torchaudio demo - move params to **kwargs in config file if works
        self.resampler = T.Resample(saved_rate, new_rate, lowpass_filter_width=64,
                           rolloff=0.9475937167399596, 
                           resampling_method="kaiser_window",beta=14.769656459379492, dtype=torch.float32)
        self.extract_fn = torchaudio.compliance.kaldi.fbank if mode == "fbank" else torchaudio.compliance.kaldi.mfcc
        self.num_mel_bins = num_mel_bins
        self.kwargs = kwargs

    def forward(self, filepath):
        waveform, sample_rate = torchaudio.load(filepath)
        y = self.resampler(waveform)
        y = self.extract_fn(y,
                            num_mel_bins=self.num_mel_bins,
                            channel=-1,
                            sample_frequency=sample_rate,
                            **self.kwargs)
        return y.transpose(0, 1).unsqueeze(0).detach()

    def extra_repr(self):
        return "mode={}, num_mel_bins={}".format(self.mode, self.num_mel_bins)
    
    
def create_transform(audio_config):
    feat_type = audio_config.pop("feat_type")
    feat_dim = audio_config.pop("feat_dim")
    
    delta_order = audio_config.pop("delta_order", 0)
    delta_window_size = audio_config.pop("delta_window_size", 2)
    apply_cmvn = audio_config.pop("apply_cmvn")
    
    is_wav = audio_config.pop("is_wav") if 'is_wav' in audio_config else False
    
    resample = audio_config.pop("resample") if 'resample' in audio_config else False
    if resample:
        saved_rate = audio_config.pop("saved_rate")
        new_rate = audio_config.pop("new_rate")

    if is_wav:
        transforms = [ExtractMelFromWav(feat_type, feat_dim, **audio_config)]
    elif resample:
        transforms = [ExtractAndResampleFeature(feat_type, feat_dim, saved_rate, new_rate, **audio_config)]
    else:
        transforms = [ExtractAudioFeature(feat_type, feat_dim, **audio_config)]

    if delta_order >= 1:
        transforms.append(Delta(delta_order, delta_window_size))

    if apply_cmvn:
        transforms.append(CMVN())

    transforms.append(Postprocess())

    return nn.Sequential(*transforms), feat_dim * (delta_order + 1)
