# Contains arguments for generating input audio representations
# These can be used either as part of the transformation or as
# part of the preprocessing (which will be included in the graph
# for gradient computation)
import sys
sys.path.append('/om/user/imgriff/python-packages/chcochleagram')
import chcochleagram

log_mel_spec_0 = {'rep_type': 'mel_spec',
                  'rep_kwargs': {'sample_rate':20000,
                                 'n_mels':256,
                                 'win_length': 1200,
                                 'hop_length': 100,
                                 'n_fft': 1200,
                                 'f_min':50,
                                 'f_max':10000},
                 'compression_type': 'log',
                 'compression_kwargs': {'offset':1e-6},
                 }

log_mel_spec_2 = {'rep_type': 'mel_spec',
                  'rep_kwargs': {'sample_rate':16000,
                                 'n_mels':48,
                                 'win_length': 320,
                                 'hop_length': 160,
                                 'n_fft': 320,
                                 'f_min':50,
                                 'f_max':10000},
                 'compression_type': 'log',
                 'compression_kwargs': {'offset':1e-6},
                 }


log_mel_spec_1 = {'rep_type': 'mel_spec',
                  'rep_kwargs': {'sample_rate':1600,
                                 'n_mels':161,
                                 'win_length': 320, # same as deepspeech; win_length = .02 * sample_rate
                                 'hop_length': 160, # same as deepspeech; hop_lenght = .01 * sample_rate
                                 'n_fft': 320, # same as deepspeech; n_fft = win_length
                                 'f_min':50,
                                 'f_max':8000},
                 'compression_type': 'log',
                 'compression_kwargs': {'offset':1e-6},
                 }

mel_spec_0 = {'rep_type': 'mel_spec',
              'rep_kwargs': {'sample_rate':20000,
                             'n_mels':256,
                             'win_length': 1200,
                             'hop_length': 100,
                             'n_fft': 1200,
                             'f_min':50,
                             'f_max':10000},
             'compression_type': 'none',
             'compression_kwargs': {'offset':1e-6},
             }

cochleagram_0 = {'rep_type': 'cochleagram',
                 'rep_kwargs': {'signal_size':2**15,
                                'sr':20000,
                                'env_sr': 200,
                                'pad_factor':None,
                                'use_rfft':True,
                                'coch_filter_type': chcochleagram.cochlear_filters.ERBCosFilters,
                                'coch_filter_kwargs': {
                                    'n':40,
                                    'low_lim':50,
                                    'high_lim':10000,
                                    'sample_factor':4,
                                    'include_highpass':False,
                                    'include_lowpass':False,
                                    'full_filter':False,
                                    },
                                'env_extraction_type': chcochleagram.envelope_extraction.RectifySubbands,
                                'downsampling_type': chcochleagram.downsampling.SincWithKaiserWindow,
                                'downsampling_kwargs': {
                                    'window_size':1001},
                               },
                 'compression_type': 'coch_p3',
                 'compression_kwargs': {'scale': 1,
                                        'offset':1e-8,
                                        'clip_value': 100,
                                        'power': 0.3}
                }

cochleagram_0 = {'rep_type': 'cochleagram',
                 'rep_kwargs': {'signal_size':2**15,
                                'sr':20000,
                                'env_sr': 200,
                                'pad_factor':None,
                                'use_rfft':True,
                                'coch_filter_type': chcochleagram.cochlear_filters.ERBCosFilters,
                                'coch_filter_kwargs': {
                                    'n':40,
                                    'low_lim':50,
                                    'high_lim':10000,
                                    'sample_factor':4,
                                    'include_highpass':False,
                                    'include_lowpass':False,
                                    'full_filter':False,
                                    },
                                'env_extraction_type': chcochleagram.envelope_extraction.RectifySubbands,
                                'downsampling_type': chcochleagram.downsampling.SincWithKaiserWindow,
                                'downsampling_kwargs': {
                                    'window_size':1001},
                               },
                 'compression_type': 'coch_p3',
                 'compression_kwargs': {'scale': 1,
                                        'offset':1e-8,
                                        'clip_value': 100,
                                        'power': 0.3}
                }


cochleagram_1 = {'rep_type': 'cochleagram',
                 'rep_kwargs': {'signal_size':40000,
                                'sr':20000,
                                'env_sr': 200,
                                'pad_factor':None,
                                'use_rfft':True,
                                'coch_filter_type': chcochleagram.cochlear_filters.ERBCosFilters,
                                'coch_filter_kwargs': {
                                    'n':50,
                                    'low_lim':50,
                                    'high_lim':10000,
                                    'sample_factor':4,
                                    'include_highpass':False,
                                    'include_lowpass':False,
                                    'full_filter':False,
                                    },
                                'env_extraction_type': chcochleagram.envelope_extraction.HilbertEnvelopeExtraction,
                                'downsampling_type': chcochleagram.downsampling.SincWithKaiserWindow,
                                'downsampling_kwargs': {
                                    'window_size':1001},
                               },
                 'compression_type': 'coch_p3',
                 'compression_kwargs': {'scale': 1,
                                        'offset':1e-8,
                                        'clip_value': 5, # This wil clip cochleagram values < ~0.04
                                        'power': 0.3}
                }


cochleagram_2 = {'rep_type': 'cochleagram',
                 'rep_kwargs': {'signal_size':32000,
                                'sr':16000,
                                'env_sr': 200,
                                'pad_factor':None,
                                'use_rfft':True,
                                'coch_filter_type': chcochleagram.cochlear_filters.ERBCosFilters,
                                'coch_filter_kwargs': {
                                    'n':50,
                                    'low_lim':50,
                                    'high_lim':8000,
                                    'sample_factor':4,
                                    'include_highpass':False,
                                    'include_lowpass':False,
                                    'full_filter':False,
                                    },
                                'env_extraction_type': chcochleagram.envelope_extraction.HilbertEnvelopeExtraction,
                                'downsampling_type': chcochleagram.downsampling.SincWithKaiserWindow,
                                'downsampling_kwargs': {
                                    'window_size':1001},
                               },
                 'compression_type': 'coch_p3',
                 'compression_kwargs': {'scale': 1,
                                        'offset':1e-8,
                                        'clip_value': 5, # This wil clip cochleagram values < ~0.04
                                        'power': 0.3}
                }

cochlea_filt = {'rep_type': 'cochlea_filt',
                 'rep_kwargs': {'sr':16000,
                                'env_sr': 8000,
                                'use_pad':True,
                                'coch_filter_path':'/om4/group/mcdermott/user/imgriff/projects/cocktail_party/deepspeech/cochlear_filter_impulse_response_16k_taps_10ms_hann_win.mat',
                                'coch_filter_kwargs': {
                                    'n_channels':50,
                                    'low_lim':40,
                                    'high_lim':8000,
                                    'sample_factor':1,
                                    'include_highpass':False,
                                    'include_lowpass':False,
                                    'full_filter':False,
                                    },
                                'env_extraction_type': 'Half-wave Rectification',
                                'downsampling_type': chcochleagram.downsampling.SincWithKaiserWindow,
                                'downsampling_kwargs': {
                                    'window_size':1001},
                               },
                 'compression_type': 'coch_p3',
                 'compression_kwargs': {'scale': 1,
                                        'offset':1e-8,
                                        'clip_value':5, # This wil clip cochleagram values < ~0.04
                                        'power': 0.3}
                }


AUDIO_INPUT_REPRESENTATIONS = {'log_mel_spec_0': log_mel_spec_0,
                               'log_mel_spec_1': log_mel_spec_1,
                               'log_mel_spec_2': log_mel_spec_2, 
                               'mel_spec_0': mel_spec_0,
                               'cochleagram_0': cochleagram_0,
                               'cochleagram_1': cochleagram_1,
                               'cochleagram_2': cochleagram_2,
                               'cochlea_filt': cochlea_filt}
