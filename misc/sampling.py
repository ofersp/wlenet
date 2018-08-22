import numpy as np


def sample_disk_uniform(n, x0, y0, radius):

    r = radius * np.sqrt(np.random.rand(n))
    theta = 2*np.pi * np.random.rand(n)
    x = x0 + r * np.cos(theta)
    y = y0 + r * np.sin(theta)
    return x, y
    

def sample_image(im, x, y):

    rx = (x + 0.5).astype('int32')
    ry = (y + 0.5).astype('int32')
    inds = np.ravel_multi_index((ry, rx), im.shape)
    return im.ravel()[inds]