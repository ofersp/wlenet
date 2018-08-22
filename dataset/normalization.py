import math as ma
import numpy as np


def norm_mean_std(stamps, axis=(1, 2)):
    stamps = stamps - np.mean(stamps, axis=axis, keepdims=True)
    stamps = stamps / np.std(stamps, axis=axis, keepdims=True)
    return stamps


def norm_sum(stamps, axis=(1, 2)):
    stamps = stamps / np.sum(stamps, axis=axis, keepdims=True)
    return stamps


def norm_max(stamps, axis=(1, 2)):
    stamps = stamps / np.max(stamps, axis=axis, keepdims=True)
    return stamps


def norm_cutoff(stamps, cutoff, axis=(1, 2)):
    stamps = norm_max(np.maximum(0.0, stamps - cutoff), axis)
    return stamps


def bg_sub(stamps):
    vert_means = (np.mean(stamps[:, :, 0], axis=1) + np.mean(stamps[:, :, -1], axis=1)) / 2
    horz_means = (np.mean(stamps[:, 0, :], axis=1) + np.mean(stamps[:, -1, :], axis=1)) / 2
    means = (vert_means + horz_means) / 2
    means = np.reshape(means, (-1, 1, 1))
    stamps = stamps - means
    return stamps, means


def norm_mean_sb(stamps):
    num_fg_pixels = np.sum((stamps > 0.0).astype('int32'), axis=(1,2), keepdims=True)
    sb = np.sum(stamps, axis=(1,2), keepdims=True) / num_fg_pixels
    return stamps / sb