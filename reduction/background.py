import numpy as np

from os.path import expanduser
from photutils import Background2D, SigmaClip, MedianBackground
from wlenet import config


def estimate(image, sigma=3.0, sigma_clip_iters=10, 
    mesh_size=(75, 75), filter_size=(5, 5), mask_value=None):

    mask = (image == mask_value)

    sigma_clip = SigmaClip(sigma=sigma, iters=sigma_clip_iters)
    bkg_estimator = MedianBackground()    
    bkg = Background2D(image, mesh_size, filter_size=filter_size,
        sigma_clip=sigma_clip, bkg_estimator=bkg_estimator, mask=mask)

    return bkg.background_median, bkg.background_rms_median, bkg, mask