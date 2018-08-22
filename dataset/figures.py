import math as ma
import numpy as np
import matplotlib.pyplot as plt
from wlenet.dataset.normalization import norm_mean_std


def show_stamps(stamps, num_pages=None, rows=10, cols=10, norm_func=norm_mean_std, figsize=[12, 12],
                xlabel=None, ylabel=None, clim=[-2, 2], no_show=False, disable_ticks=True):

    stamps = norm_func(stamps) if not norm_func is None else stamps

    stamp_sz = stamps.shape[1:3]
    stamps = stamps.reshape((-1,stamp_sz[0],stamp_sz[1],1))

    stamps_per_page = rows * cols
    num_pages_ = int(ma.ceil(stamps.shape[0] / stamps_per_page))
    num_pages = num_pages_ if num_pages is None else min(num_pages_, num_pages)

    j = 0
    for i in range(num_pages):
        page = np.zeros(np.array(stamp_sz)*np.array([rows, cols]))
        for k in range(rows):
            for l in range(cols):
                if j<stamps.shape[0]:
                    page[k*stamp_sz[0]:(k+1)*stamp_sz[0],
                         l*stamp_sz[1]:(l+1)*stamp_sz[1]] = stamps[j, :, :, 0]
                j += 1

        if figsize is not None:
            fig = plt.figure(figsize=figsize)

        if clim is None:
            plt.imshow(page)
        else:
            plt.imshow(page, clim=clim)

        if disable_ticks:
            plt.xticks([], [])
            plt.yticks([], [])

        if xlabel is not None:
            plt.xlabel(xlabel)
        if ylabel is not None:
            plt.ylabel(ylabel)

        if not no_show:
            plt.show()