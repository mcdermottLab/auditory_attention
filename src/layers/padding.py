""" Padding Helpers
Hacked together by Ross Wightman
"""
import math
from typing import List, Tuple

import torch.nn.functional as F


# Calculate symmetric padding for a convolution
def get_padding(kernel_size: int, stride: int = 1, dilation: int = 1, **_) -> int:
    if isinstance(kernel_size, list):
        stride_h, stride_w = stride if isinstance(stride, list) else [stride, stride]
        dilation_h, dilation_w = dilation if isinstance(dilation, list) else [dilation, dilation]
        kernel_h, kernel_w = kernel_size if isinstance(kernel_size, list) else [kernel_size, kernel_size]
        h_pad = ((stride_h - 1) + dilation_h * (kernel_h - 1)) // 2
        w_pad = ((stride_w - 1) + dilation_w * (kernel_w - 1)) // 2
        return [h_pad, w_pad]

    padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
    return padding


# Calculate asymmetric TensorFlow-like 'SAME' padding for a convolution
def get_same_padding(x: int, k: int, s: int, d: int):
    return max((math.ceil(x / s) - 1) * s + (k - 1) * d + 1 - x, 0)


# Can SAME padding for given args be done statically?
def is_static_pad(kernel_size: int, stride: int = 1, dilation: int = 1, **_):
    if isinstance(kernel_size, list):
        return stride == 1 and (dilation * (kernel_size[0] - 1)) % 2 == 0 and (dilation * (kernel_size[1] - 1)) % 2 == 0
    return stride == 1 and (dilation * (kernel_size - 1)) % 2 == 0


# Dynamically pad input x with 'SAME' padding for conv with specified args
def pad_same(x, k: List[int], s: List[int], d: List[int] = (1, 1), value: float = 0):
    ih, iw = x.size()[-2:]
    pad_h, pad_w = get_same_padding(ih, k[0], s[0], d[0]), get_same_padding(iw, k[1], s[1], d[1])
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2], value=value)
    return x

def pad_same1d(x, k: List[int], s: List[int], d: List[int] = (1, 1), value: float = 0):
    ih = x.size()[-1]
    pad_h = get_same_padding(ih, k[0], s[0], d[0])
    if pad_h > 0 :
        x = F.pad(x, [pad_h // 2, pad_h - pad_h // 2], value=value)
    return x


def get_padding_value(padding, kernel_size, **kwargs) -> Tuple[Tuple, bool]:
    dynamic = False
    if isinstance(padding, str):
        # for any string padding, the padding will be calculated for you, one of three ways
        padding = padding.lower()
        if padding == 'same':
            # TF compatible 'SAME' padding, has a performance and GPU memory allocation impact
            if is_static_pad(kernel_size, **kwargs):
                # static case, no extra overhead
                padding = get_padding(kernel_size, **kwargs)
            else:
                # dynamic 'SAME' padding, has runtime/GPU memory overhead
                padding = 0
                dynamic = True
        elif padding == 'valid':
            # 'VALID' padding, same as padding=0
            padding = 0
        elif padding == 'valid_time':
            padding = get_padding(kernel_size, **kwargs)
            padding[1] = 0
        else:
            # Default to PyTorch style 'same'-ish symmetric padding
            padding = get_padding(kernel_size, **kwargs)
    return padding, dynamic