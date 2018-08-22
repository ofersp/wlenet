import math as ma
import numpy as np

from skimage.util import view_as_blocks
from astropy.convolution import convolve
from scipy.signal import convolve2d


def correlated_noise_stamps(num_stamps, stamp_sz, noise_kernel, seed, 
                            aug_noise_factor, aug_noise_factor_min, aug_noise_factor_max):

    stamp_sz = np.array(stamp_sz)
    kernel_sz = np.array(noise_kernel.shape)
    pad_sz = np.ceil(kernel_sz / 2.0).astype('int32') + 1
    padded_stamp_sz = stamp_sz + 2 * pad_sz
    noise_im_sz = padded_stamp_sz * int(ma.ceil(ma.sqrt(num_stamps)))

    np.random.seed(seed)
    noise_im = np.random.randn(*noise_im_sz)
    noise_im = convolve(noise_im, noise_kernel, normalize_kernel=False, boundary=None) 
    
    noise_stamps = view_as_blocks(noise_im, tuple(padded_stamp_sz)).reshape(-1, *padded_stamp_sz).copy()
    noise_stamps = noise_stamps[:num_stamps, pad_sz[0]:-pad_sz[0], pad_sz[1]:-pad_sz[1]]
    noise_stamps = noise_stamps.reshape(-1, stamp_sz[0], stamp_sz[1], 1)

    if aug_noise_factor:
        aug_noise_range = aug_noise_factor_max - aug_noise_factor_min
        noise_factors = np.random.rand(num_stamps, 1, 1, 1) * aug_noise_range + aug_noise_factor_min
        noise_stamps = noise_stamps * noise_factors

    return noise_stamps