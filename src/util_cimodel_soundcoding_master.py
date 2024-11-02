import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import src.utils_ci_signal as utils_signal

class SequentialCIModel(nn.Module):
    def __init__(self, config, center_crop=True, out_dur = 2.0):
        super(SequentialCIModel, self).__init__()
        self.config = config
        self.get_spike =  SpikeGeneratorBinomial(**config['kwargs_spike_generator_binomial'])
        self.anf_out_from_in = anf_input_to_anf_output(config['kwargs_anf_output'])
        self.anf_in_from_pulse = pulsetrains_to_anf_input(**config['kwargs_anf_input'])
        self.pulse_from_current = currentmap_to_pulsetrains(config['sr_coch'],**config['kwargs_pulsetrains'])
        self.current_from_pressure = sigmoid_current_map(**config['kwargs_currentmap_sigmoid'])
        self.env_from_subband = subbands_to_envelopes(sr_input=config['sr_audio'],sr_output=config['sr_coch'],**config['kwargs_envelopes'])
        self.subband_from_audio = audio_to_subbands(sr=self.config['sr_audio'],**config['kwargs_subbands'])
        self.n_out_frames = int(out_dur * config['sr_coch'])
        self.center_crop = center_crop

    def forward(self, x):
        # print('Input shape:', x.shape)
        if self.config.get('kwargs_subbands', False):
            x = self.subband_from_audio(x)
            # print('Subbands shaepe:', x.shape)
        
        if self.config.get('kwargs_envelopes', False):
            x = self.env_from_subband(x)
            # print('Envelopes shaepe:', x.shape)
        
        if self.config.get('kwargs_currentmap_sigmoid', False):
            x = self.current_from_pressure(x)
            # print('Compressed envelope shaepe:', x.shape)
        
        if self.config.get('kwargs_pulsetrains', False):
            x = self.pulse_from_current(x)
            # print('Pulsetrain shaepe:', x.shape)
        
        if self.config.get('kwargs_anf_input', False):
            x = self.anf_in_from_pulse(x)
            # print('ANF stimulation shaepe:', x.shape)
        
        if self.config.get('kwargs_anf_output', False):
            x = self.anf_out_from_in(x)
            # print('ANF firing response shaepe:', x.shape)

        if self.config.get('kwargs_spike_generator_binomial', False):
            x = self.get_spike(x).permute(1,0,2,3)
            # print('Final nervegram shaepe:', x.shape)

        if self.center_crop:
            x_dur = x.shape[-1]
            diff = (x_dur - self.n_out_frames) // 2
            frame_start = diff 
            frame_end = int(x_dur - diff)
            x = x[..., frame_start : frame_end]
        return x.float()

class audio_to_subbands(nn.Module):
    def __init__(self, sr=32000, rectify=True, **kwargs_subbands):
        super(audio_to_subbands, self).__init__()
        self.sr = sr
        self.rectify = rectify
        self.config_filterbank = kwargs_subbands
        self.signal_length = self.config_filterbank.get('signal_length', 110_250)
        self.filterbank_mode = self.config_filterbank.pop('mode', None)
        # register buffer
        self.register_buffer('filterbank', self.get_filterbank())
        # Convert audio to subbands as specified by config_filterbank
        print('[cimodel] converting audio to subbands using {}'.format(self.filterbank_mode))

    def get_filterbank(self):
        # Prepare list of CFs / BWs (default is spaced linearly on an ERB scale)
        min_lo = self.config_filterbank.get('min_lo', 80.0)
        max_hi = self.config_filterbank.get('max_hi', 8000.0)
        num_cf = self.config_filterbank.get('num_cf', 50)
        list_cutoffs = utils_signal.erbspace(min_lo, max_hi, num_cf + 2)
        list_lo = list_cutoffs[:-2]
        list_hi = list_cutoffs[2:]
        cfs = list_cutoffs[1:-1]
        bws = list_hi - list_lo
        bws_erb = utils_signal.freq2erb(list_hi) - utils_signal.freq2erb(list_lo)
        # Prepare input signal and apply half-cosine filterbank in frequency domain
        filts, freqs = utils_signal.make_half_cosine_filters(
            self.signal_length,
            self.sr,
            list_lo=list_lo,
            list_hi=list_hi)
        rfft_filts = torch.unsqueeze(torch.tensor(filts, dtype=torch.complex64),0)
        return rfft_filts

    def apply_freq_domain_filterband(self, x):
        # Prepare input signal and apply roex filterbank in frequency domain
        if len(x.shape) == 1:
            x = torch.unsqueeze(x,0)
        elif len(x.shape) > 2:
            raise ValueError("Input dimensions should be: [batch, time]")
        rfft_x = torch.fft.rfft(x.to(torch.float32))
        rfft_x = torch.unsqueeze(rfft_x,1)
        rfft_y = torch.multiply(rfft_x, self.filterbank)
        y = torch.fft.irfft(rfft_y)
        return y

    def forward(self, tensor_audio):
        if self.filterbank_mode is None:
            assert not self.config_filterbank, "filterbank_mode must be specified in config_filterbank"
            assert len(tensor_audio.shape) >= 3, "shape must be [batch, time, freq, (channel)] to skip filterbank"
        elif self.filterbank_mode == 'half_cosine_filterbank':
            # Apply half-cosine filterbank in frequency domain
            tensor_subbands = self.apply_freq_domain_filterband(tensor_audio)

        # Half-wave rectify subbands
        if self.rectify:
            tensor_subbands = F.relu(tensor_subbands)

            return tensor_subbands

class subbands_to_envelopes(nn.Module):
    def __init__(self, sr_input=20e3, sr_output=10e3,
                 envelope_lowpass_ideal=False, kwargs_fir_lowpass_filter={}):
        super(subbands_to_envelopes, self).__init__()
        
        self.sr_input = sr_input
        self.sr_output = sr_output
        self.envelope_lowpass_ideal = envelope_lowpass_ideal
        self.kwargs_fir_lowpass_filter = kwargs_fir_lowpass_filter
        self.resample = tf_fir_resample(
                sr_input=self.sr_input,
                sr_output=self.sr_output,
                kwargs_fir_lowpass_filter=self.kwargs_fir_lowpass_filter,
                return_io_function=False
            )

    def forward(self, tensor_subbands):
        # Half-wave rectify for envelope extraction
        tensor_envelopes = F.relu(tensor_subbands)
        
        if self.envelope_lowpass_ideal:
            # print("[cimodel] Performing subband envelope extraction with ideal lowpass filter")
            
            # If an ideal lowpass filter is needed, extract envelopes by downsampling
            # to 2 x envelope_lowpass_cutoff, and then upsampling to sr_output
            tensor_envelopes = self.resample(tensor_envelopes)
            # Half-wave rectify once more to eliminate negative artifacts from envelopes
            tensor_envelopes = F.relu(tensor_envelopes)
        
        return tensor_envelopes

class sigmoid_current_map(nn.Module):
    def __init__(self, TL=-5, MCL=5, threshold=0.0, dynamic_range=25.0, dynamic_range_interval=0.95):
        super(sigmoid_current_map, self).__init__()
        
        # Convert TL, MCL, threshold, and dynamic_range to the required form
        rate_spont = 1e-3 * 10 ** (TL / 20)
        rate_max = 1e-3 * 10 ** (MCL / 20)
        rate_spont = np.array(rate_spont).reshape([-1])
        rate_max = np.array(rate_max).reshape([-1])
        self.threshold = torch.tensor(threshold).reshape([-1])
        self.dynamic_range = torch.tensor(dynamic_range).reshape([-1])
        
        # Ensure valid arguments
        assert np.all(rate_max > rate_spont), "rate_max must be greater than rate_spont"
        self.argument_lengths = [len(rate_spont), len(rate_max), len(self.threshold), len(self.dynamic_range)]
        self.n_channels = max(self.argument_lengths)
        channel_specific_shape = [1, 1, 1, self.n_channels]
        msg = "inconsistent argument lengths for rate_spont, rate_max, threshold, dynamic_range"
        assert np.all([_ in (1, self.n_channels) for _ in self.argument_lengths]), msg
        
        # Constants for sigmoid calculation
        self.y_threshold = torch.tensor((1 - dynamic_range_interval) / 2).view(channel_specific_shape)
        k = (torch.log((1 / self.y_threshold) - 1) / (self.dynamic_range / 2)).view(channel_specific_shape)
        x0 = (self.threshold - (torch.log((1 / self.y_threshold) - 1) / (-k))).view(channel_specific_shape)
        rate_spont = torch.from_numpy(rate_spont).view(channel_specific_shape)
        rate_max = torch.from_numpy(rate_max).view(channel_specific_shape)
        self.log_of_10 = np.log(10)
        
        # register buffer
        self.register_buffer('rate_spont', rate_spont)
        self.register_buffer('rate_max', rate_max)
        self.register_buffer('k', k)
        self.register_buffer('x0', x0)

        
    def forward(self, X):        
        # Handle 3D or 4D inputs
        if len(X.shape) == 4:
            assert X.shape[-1] in (1, self.n_channels), "Number of channels in tensor_subbands must be 1 or {}".format(self.n_channels)
        if len(X.shape) == 3:
            if self.n_channels != 1:
                X = X.unsqueeze(-1)  # Add a channel dimension if necessary
        
        # # Convert numpy arrays to torch tensors, matching the input dtype
        # rate_spont = torch.tensor(self.rate_spont, dtype=X.dtype).view(channel_specific_shape)
        # rate_max = torch.tensor(self.rate_max, dtype=X.dtype).view(channel_specific_shape)
        # k = torch.tensor(self.k, dtype=X.dtype).view(channel_specific_shape)
        # x0 = torch.tensor(self.x0, dtype=X.dtype).view(channel_specific_shape)
        
        # Compute the sigmoid function in PyTorch
        x = 20.0 * torch.log(X / 20e-6) / self.log_of_10
        y = 1.0 / (1.0 + torch.exp(-self.k * (x - self.x0)))
        
        # Compute the current map
        current_map = self.rate_spont + (self.rate_max - self.rate_spont) * y
        return current_map.squeeze(0)


class currentmap_to_pulsetrains(nn.Module):
    def __init__(self, sr=20e3, pps=500.0, offset=0.0, compression_power=0.3, num_elec=16, num_samples=25000):
        super(currentmap_to_pulsetrains, self).__init__()
        self.sr = sr  # sampling rate
        self.pps = pps  # pulses per second
        self.offset = offset  # offset in seconds
        self.compression_power = compression_power  # compression exponent
        self.ipi_in_seconds = 1.0 / self.pps  # inter-pulse interval in seconds
        self.ipi_in_samples = int(self.ipi_in_seconds * self.sr)  # inter-pulse interval in samples
        offset_in_samples = int(self.offset * self.sr)  # offset in samples
        

        pulse_mask_array = self.get_pulse_mask(
            num_elec=num_elec,
            num_samples= num_samples + self.ipi_in_samples,
            ipi=self.ipi_in_samples,
            offset=offset_in_samples
        ).unsqueeze(0)
        self.register_buffer('pulse_mask_array', pulse_mask_array)

    def get_pulse_mask(self, num_elec, num_samples, ipi=1, offset=0):
        """
        Helper function returns a binary mask with shape (num_elec, num_samples)
        containing ones at intervals specified by ipi and offset
        """
        mask = torch.zeros((num_elec, num_samples), dtype=torch.int)
        for itr_elec in range(num_elec):
            tmp = (offset * itr_elec) % ipi
            mask[itr_elec, tmp:-1:ipi] = 1
        return mask

    def forward(self, tensor_currentmap):        
        # Randomly select the start time for the pulse mask
        t_start = torch.randint(
            low=0,
            high=self.ipi_in_samples,
            size=(),
            dtype=torch.int64
        )
        t_end = t_start + tensor_currentmap.shape[2]
        
        # Apply the pulse mask via pointwise multiplication
        tensor_pulsetrains = tensor_currentmap * self.pulse_mask_array[:, :, t_start:t_end]
        
        # Apply compression if compression_power is provided
        if self.compression_power is not None:
            tensor_pulsetrains = torch.pow(tensor_pulsetrains, self.compression_power).float()
        
        return tensor_pulsetrains
    
class pulsetrains_to_anf_input(nn.Module):
    def __init__(self, num_anfs=100, elec_min_x=8.125, elec_max_x=23.875, anfs_min_x=None, anfs_max_x=None, anfs_min_cf=125.0, anfs_max_cf=14e3, verbose=False, num_elec=16, kwargs_current_spread={}):
        super(pulsetrains_to_anf_input, self).__init__()
        self.num_anfs = num_anfs  # number of auditory nerve fibers
        self.elec_min_x = elec_min_x  # electrode position in cochlea
        self.elec_max_x = elec_max_x  # electrode position in cochlea
        self.anfs_min_x = anfs_min_x  # spread of the ANRs in the cochlea
        self.anfs_max_x = anfs_max_x  # spread of the ANRs in the cochlea
        self.anfs_min_cf = anfs_min_cf  # minimum ANF frequency
        self.anfs_max_cf = anfs_max_cf  # maximum ANF frequency
        self.kwargs_current_spread = kwargs_current_spread
        self.num_elec = num_elec
        self.verbose = verbose  # verbose
        self.register_buffer("tensor_current_spread", self.get_nerve_interface().float()) 
        
    def get_current_spread_matrix(self, elec_x,
                              anfs_x,
                              V0=1.0,
                              lambda_current_spread=1.0):
        """
        Helper function to calculate the matrix that converts from the electrode current
        to the effective current experienced by each auditory nerve fiber.
        
        Note that the defaults are based on (Busby. et al 1994):
        Pitch perception for different modes of stimulation using the
        cochlear multiple electrode prosthesis. J. Acoust. Soc. Am.
        
        They modeled a 22-electrode cochlear implant developed by Cochlear Pty. Limited.
        """
        # To increase current spread, increase lambda. 
        # Calculate the current spread function
        [ELEC_X, ANFS_X] = torch.meshgrid(elec_x.float(), anfs_x.float())
        V = V0 * torch.exp(-torch.abs(ANFS_X - ELEC_X) / lambda_current_spread)
        return V
    
    def get_nerve_interface(self):
        elec_x = torch.linspace(self.elec_min_x, self.elec_max_x, self.num_elec) # mm from apex

        if self.verbose:
            tmp = "[cimodel] {} electrodes spaced linearly from {} to {} mm from apex"
            print(tmp.format(self.num_elec, self.elec_min_x, self.elec_max_x))

        # Determine positions of the ANFs along the basilar membrane
        anfs_cfs = torch.tensor(utils_signal.erbspace(self.anfs_min_cf, self.anfs_max_cf, self.num_anfs))
        anfs_x = (35 / 2.1) * torch.log10(1.0 + anfs_cfs / 165.4)

        if self.verbose:
            print("[cimodel] Determining ANF positions based on ERB-spaced CFs")
            tmp = "[cimodel] {} ANFs spaced linearly from {:.4f} to {:.4f} mm from apex"
            print(tmp.format(self.num_anfs, anfs_x[0], anfs_x[-1]))
            tmp = "[cimodel] {} ANFs with CFs ERB-spaced from {:.4f} to {:.4f} Hz"
            print(tmp.format(self.num_anfs, anfs_cfs[0], anfs_cfs[-1]))
        
        V = self.get_current_spread_matrix(
            elec_x=elec_x,
            anfs_x=anfs_x,
            **self.kwargs_current_spread)
        V = V.permute(1,0).unsqueeze(0)
        print('tensor_current_spread shape:', V.shape)
        return V
    
    def forward(self, tensor_pulsetrains):
        tensor_anf_input = torch.matmul(self.tensor_current_spread, tensor_pulsetrains)
        return tensor_anf_input
    

class anf_input_to_anf_output(nn.Module):
    def __init__(self, kwargs_anf_output={}):
        super(anf_input_to_anf_output, self).__init__()
        self.kwargs_sigmoid_rate_level_function = kwargs_anf_output['kwargs_sigmoid_rate_level_function']
        self.kwargs_anf_impulse_response = kwargs_anf_output['kwargs_anf_impulse_response']
        kernel = self.get_anf_impulse_response(**self.kwargs_anf_impulse_response)
        kernel = kernel.flip(0).unsqueeze(0).unsqueeze(0).unsqueeze(0)  # Reverse kernel and reshape
        self.register_buffer('kernel', kernel)

        if self.kwargs_sigmoid_rate_level_function:
            self.rate_level_function = SigmoidRateLevelFunction(**self.kwargs_sigmoid_rate_level_function)
            print('[cimodel] incorporated sigmoid_rate_level_function: {}'.format(self.kwargs_sigmoid_rate_level_function))
        
    def get_anf_impulse_response(self, sr=20e3,
                             kernel_dur=0.005,
                             absolute_latency=0.0,
                             exc_mean_latency=0.6e-3,
                             exc_std_frac=1.0,
                             exc_weight=1.0,
                             ref_mean_period=1.6e-3,
                             ref_std_frac=0.15,
                             ref_weight=0.1):
        '''
        Helper function to create impulse response of LTI system approximating auditory
        nerve fiber responses to amplitude modulated current impulses delivered by
        cochlear implant electrodes.
        Currently this impulse response is simply a difference of two Gaussians: one
        excitatory component and one refractory component.
        '''
        # Define the kernel time vector
        kernel_t = torch.arange(-kernel_dur, kernel_dur + 1/sr, 1/sr)
        kernel = torch.zeros_like(kernel_t)

        # Define mean and standard deviation of the excitatory and refractory components
        mu_exc = exc_mean_latency
        sigma_exc = mu_exc * exc_std_frac
        mu_ref = mu_exc + ref_mean_period
        sigma_ref = mu_ref * ref_std_frac

        # Calculate impulse response as a difference of Gaussians with arbitrary re-scaling
        pidx = kernel_t > absolute_latency

        # Compute the excitatory and refractory components using PyTorch
        exc_gaussian = exc_weight * torch.exp(-0.5 * ((kernel_t[pidx] - mu_exc) / sigma_exc) ** 2) / (sigma_exc * torch.sqrt(torch.tensor(2 * torch.pi)))
        ref_gaussian = ref_weight * torch.exp(-0.5 * ((kernel_t[pidx] - mu_ref) / sigma_ref) ** 2) / (sigma_ref * torch.sqrt(torch.tensor(2 * torch.pi)))

        # Apply the difference of Gaussians
        kernel[pidx] += exc_gaussian
        kernel[pidx] -= ref_gaussian

        # Normalize the kernel
        kernel = kernel / torch.max(kernel)
        return kernel

    def forward(self, tensor_anf_input):
        # Convert kernel to PyTorch tensor with the same dtype as tensor_anf_input
        # tensor_kernel = torch.tensor(kernel, dtype=tensor_anf_input.dtype, device=tensor_anf_input.device)

        # Reshape input and kernel tensors for the 2D convolution
        tensor_anf_input_reshaped = tensor_anf_input.unsqueeze(1)  # Add channel dimension
        # tensor_kernel_reshaped = tensor_kernel.flip(0).unsqueeze(0).unsqueeze(0).unsqueeze(0)  # Reverse kernel and reshape
        # tensor_kernel_reshaped = self.tensor_kernel.flip(0).unsqueeze(0).unsqueeze(0).unsqueeze(0)  # Reverse kernel and reshape

        # Perform the 2D convolution
        tensor_anf_output = F.conv2d(
            tensor_anf_input_reshaped,
            self.kernel,
            padding='same'  # Equivalent to 'SAME' padding in TensorFlow
        )

        # Apply ReLU activation
        tensor_anf_output = F.relu(tensor_anf_output.squeeze(1))  # Remove the channel dimension

        # Convert from amplitude / sound level units to ANF spike rates
        if self.kwargs_sigmoid_rate_level_function:
            tensor_output_ratelevel = self.rate_level_function(tensor_anf_output)
            return tensor_output_ratelevel
        
        return tensor_anf_output


class SigmoidRateLevelFunction(nn.Module):
    def __init__(self, rate_spont=70.0, rate_max=250.0, threshold=0.0, dynamic_range=25.0, dynamic_range_interval=0.95):
        super(SigmoidRateLevelFunction, self).__init__()
        
        # Convert rate_spont, rate_max, threshold, and dynamic_range to the required form
        rate_spont = np.array(rate_spont).reshape([-1])
        rate_max = np.array(rate_max).reshape([-1])
        self.threshold = np.array(threshold).reshape([-1])
        self.dynamic_range = np.array(dynamic_range).reshape([-1])
        
        # Constants for sigmoid calculation
        self.y_threshold = (1 - dynamic_range_interval) / 2
        k = np.log((1 / self.y_threshold) - 1) / (self.dynamic_range / 2)
        x0 = self.threshold - (np.log((1 / self.y_threshold) - 1) / (-k))
        
        # Ensure valid arguments
        assert np.all(rate_max > rate_spont), "rate_max must be greater than rate_spont"
        self.argument_lengths = [len(rate_spont), len(rate_max), len(self.threshold), len(self.dynamic_range)]
        self.n_channels = max(self.argument_lengths)
        msg = "Inconsistent argument lengths for rate_spont, rate_max, threshold, dynamic_range"
        assert np.all([_ in (1, self.n_channels) for _ in self.argument_lengths]), msg
        channel_specific_shape = [1, 1, 1, self.n_channels]

        # register buffer
        self.register_buffer('rate_spont', torch.tensor(rate_spont).view(channel_specific_shape))
        self.register_buffer('rate_max', torch.tensor(rate_max).view(channel_specific_shape))
        self.register_buffer('k', torch.tensor(k).view(channel_specific_shape))
        self.register_buffer('x0', torch.tensor(x0).view(channel_specific_shape))

        self.log_of_10 = np.log(10)

    def forward(self, tensor_subbands):
        # Handle 3D or 4D inputs
        if len(tensor_subbands.shape) == 4:
            assert tensor_subbands.shape[-1] in (1, self.n_channels), \
                f"Number of channels in tensor_subbands must be 1 or {self.n_channels}"
        if len(tensor_subbands.shape) == 3:
            if self.n_channels != 1:
                tensor_subbands = tensor_subbands.unsqueeze(-1)  # Add channel dimension if necessary

        # Compute the sigmoid function in PyTorch 
        # TODO: make this a general function - don't hardcode in each class
        x = 20.0 * torch.log(tensor_subbands / 1e-3) / self.log_of_10
        y = 1.0 / (1.0 + torch.exp(-self.k * (x - self.x0)))

        # Compute the spiking rate
        tensor_rates = self.rate_spont + (self.rate_max - self.rate_spont) * y
        return tensor_rates
    
class SpikeGeneratorBinomial(nn.Module):
    def __init__(self, sr=10000, n_per_channel=1, mode='approx', p_dtype='float32', max_prop = 0.8, min_prop = 0.05):
        super(SpikeGeneratorBinomial, self).__init__()
        self.sr = sr
        self.n_per_channel = np.array(n_per_channel, dtype=int).reshape([1, 1, 1, -1])
        self.mode = mode.lower()
        assert self.mode in ['approx', 'exact', 'gradient_approx']
        self.p_dtype = torch.float32
        self.max_prop = max_prop
        self.min_prop = min_prop
        n = torch.tensor(self.n_per_channel, dtype=self.p_dtype)
        self.register_buffer('n', n)    

    def forward(self, inputs):
        tensor_spike_probs = torch.multiply(inputs, 1 / self.sr)
        if len(tensor_spike_probs.shape) < 4:
            tensor_spike_probs = torch.unsqueeze(tensor_spike_probs, dim=-1)
        # Approx implementation: sample from normal approximation of binomial distribution
        p = tensor_spike_probs
        q = 1 - tensor_spike_probs
        mean = (self.n * p).expand_as(tensor_spike_probs)
        std = torch.sqrt(self.n * p * q).expand_as(tensor_spike_probs)
        tensor_spike_counts = torch.normal(
            mean,                # Mean of the normal distribution
            std,  # Standard deviation
        )

        # Apply ReLU activation and round the result
        tensor_spike_counts = torch.round(F.relu(tensor_spike_counts))

        # Cast tensor to the same type as `inputs`
        tensor_spike_counts = tensor_spike_counts
        return tensor_spike_counts

class tf_fir_resample(torch.nn.Module):
    def __init__(self, sr_input, sr_output, kwargs_fir_lowpass_filter={}, verbose=False, return_io_function=True):
                  
        """
        Class for resampling time-domain signals with an FIR lowpass filter.
        
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
        super(tf_fir_resample, self).__init__()
        kwargs_fir_lowpass_filter = dict(kwargs_fir_lowpass_filter) # prevents modifying in-place
        if kwargs_fir_lowpass_filter.get('cutoff', None) is None:
            kwargs_fir_lowpass_filter['cutoff'] = sr_output / 2
        if verbose:
            print('[tf_fir_resample] `kwargs_fir_lowpass_filter`: {}'.format(kwargs_fir_lowpass_filter))
        
        # Compute upsample and downsample factors
        greatest_common_divisor = np.gcd(int(sr_output), int(sr_input))
        self.up = int(sr_output) // greatest_common_divisor
        self.down = int(sr_input) // greatest_common_divisor
        filt, sr_filt = utils_signal.fir_lowpass_filter(sr_input, sr_output, **kwargs_fir_lowpass_filter, verbose=verbose)
        filt = filt * self.up # Re-scale filter to offset attenuation from upsampling
        tensor_kernel_lowpass_filter = torch.unsqueeze(torch.unsqueeze(torch.from_numpy(filt),0),0).float()
        # register buffer
        self.register_buffer('tensor_kernel_lowpass_filter', tensor_kernel_lowpass_filter)
        self.verbose = verbose  
        # First upsample by a factor of `up` by adding `up-1` zeros between each sample in original signal
        self.nzeros = self.up - 1

        pad_length = int((self.tensor_kernel_lowpass_filter.shape[-1]-1)/2)
        self.paddings = (pad_length, pad_length)

    def forward(self, tensor_input):
        # Expand dimensions of input tensor to [batch, freq, time, channels] for 2d conv operation
        if len(tensor_input.shape) == 2:
            f = 1
            t = tensor_input.shape[1]
            c = 1
            tensor_input = torch.unsqueeze(torch.unsqueeze(tensor_input,1),-1)
            if self.verbose:
                print(f'[tf_fir_resample] interpreted `tensor_input.shape` as [batch, time={t}]')
        elif len(tensor_input.shape) == 3:
            f = tensor_input.shape[1]
            t = tensor_input.shape[2]
            c = 1
            tensor_input = torch.unsqueeze(tensor_input,-1)
            if self.verbose:
                print(f'[tf_fir_resample] interpreted `tensor_input.shape` as [batch, freq={f}, time={t}]')
        else:
            msg = "dimensions of `tensor_input` must support re-shaping to [batch, freq, time, channels]"
            assert (len(tensor_input.shape) == 4), msg
            f = tensor_input.shape[1]
            t = tensor_input.shape[2]
            c = tensor_input.shape[3]
            tensor_input = tensor_input
            if self.verbose:
                print(f'[tf_fir_resample] interpreted `tensor_input.shape` as [batch, freq={f}, time={t}, channels={c}]')

        if self.nzeros > 0:
            paddings = (0,0,0,1,0,0,0,0)
            indices = []
            for _ in range(t):
                indices.append(_)
                indices.extend([t] * self.nzeros)
            tensor_input_padded = torch.nn.functional.pad(
                tensor_input,
                pad=paddings,
                mode='constant')
            # Insert up - 1 zeros between each sample
            tensor_input_padded = tensor_input_padded[:, :, indices]
        else:
            tensor_input_padded = tensor_input

        tensor_output = torch.nn.functional.pad(
            tensor_input_padded.squeeze(-1),
            pad=self.paddings,
            mode='constant') 

        tensor_input_lowpass_filtered = torch.nn.functional.conv2d(tensor_output.unsqueeze(1),
                                                    self.tensor_kernel_lowpass_filter.unsqueeze(1),
                                                    stride=(1,self.down), padding='valid')

        tensor_output = tensor_input_lowpass_filtered.squeeze(1)

        return tensor_output