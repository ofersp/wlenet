import math as ma
import numpy as np
from galsim import image as gsimage
from wlenet.dataset.normalization import norm_max


def stamp_centered_grid(sz, origin_shift=(0, 0)):

    y = np.linspace(-(sz[0]-1)/2, (sz[0]-1)/2, sz[0]) - origin_shift[0]
    x = np.linspace(-(sz[1]-1)/2, (sz[1]-1)/2, sz[1]) - origin_shift[1]
    xx, yy = np.meshgrid(x, y)
    return xx, yy


def extract(image, mask, ra_dec_pix, stamp_sz):

    mask = np.zeros(image.shape) if mask is None else mask
    stamps = np.ones((ra_dec_pix.shape[0], stamp_sz[0], stamp_sz[1]))*np.nan
    status = np.zeros((ra_dec_pix.shape[0], 1))
    lt = np.round(ra_dec_pix - np.array([stamp_sz[::-1]])/2).astype(np.int32)
    rb = lt + np.array([stamp_sz[::-1]])
    xx, yy = stamp_centered_grid(stamp_sz)

    for i in range(ra_dec_pix.shape[0]):
        curr_stamp = image[lt[i, 1]:rb[i, 1], lt[i, 0]:rb[i, 0]]
        curr_stamp_mask = mask[lt[i, 1]:rb[i, 1], lt[i, 0]:rb[i, 0]]        
        no_nans = not np.any(np.isnan(curr_stamp))
        no_mask = not np.any(curr_stamp_mask)
        proper_shape = np.all(curr_stamp.shape == stamp_sz)
        if proper_shape and no_nans and no_mask:
            stamps[i, :, :] = curr_stamp
            status[i] = 1
    return stamps, status


def rrg_shapes(stamps, weight_radii, psf_radius):

    stamps = stamps.copy()
    stamp_sz = stamps.shape[1:3]
    x, y = stamp_centered_grid(stamp_sz)
    xx = (x*x).reshape(-1, 1)
    yy = (y*y).reshape(-1, 1)
    xy = (x*y).reshape(-1, 1)    
    chi = np.zeros((len(stamps),), dtype='complex64')
    
    for i in range(len(stamps)):

        g_sqr = weight_radii[i]**2
        gw_sqr = (weight_radii[i]**-2 + psf_radius**-2)**-1

        stamp = stamps[i, ...]
        W = np.exp(-0.5*(xx + yy) / g_sqr).reshape(1, -1)
        stamp = stamp.reshape(1, -1) * W
        stamp = stamp / np.sum(stamp, axis=1, keepdims=True)

        Q_xx = (stamp.dot(xx) - gw_sqr) * (gw_sqr/g_sqr)**2 
        Q_yy = (stamp.dot(yy) - gw_sqr) * (gw_sqr/g_sqr)**2 
        Q_xy = stamp.dot(xy) * (gw_sqr/g_sqr)**2

        num = Q_xx - Q_yy + 2*(1j)*Q_xy
        chi_denom = Q_xx + Q_yy
        chi[i] = num / chi_denom
        
    chi = np.vstack((np.real(chi), np.imag(chi))).T
    return chi


def chi_shapes(stamps, weight_radii):

    stamps = stamps.copy()
    stamp_sz = stamps.shape[1:3]
    x, y = stamp_centered_grid(stamp_sz)
    xx = (x*x).reshape(-1, 1)
    yy = (y*y).reshape(-1, 1)
    xy = (x*y).reshape(-1, 1)    
    chi = np.zeros((len(stamps),), dtype='complex64')
    
    for i in range(len(stamps)):

        stamp = stamps[i, ...]
        W = np.exp(-0.5*(xx + yy) / weight_radii[i]**2).reshape(1, -1)
        stamp = stamp.reshape(1, -1) * W
        stamp = stamp / np.sum(stamp, axis=1, keepdims=True)        
        Q_xx = stamp.dot(xx)
        Q_yy = stamp.dot(yy)
        Q_xy = stamp.dot(xy)
        num = Q_xx - Q_yy + 2*(1j)*Q_xy
        chi_denom = Q_xx + Q_yy
        chi[i] = num / chi_denom
        
    chi = np.vstack((np.real(chi), np.imag(chi))).T
    return chi


def moments(stamps, weight_radius=None, zero_cutoff=False):

    stamps = stamps.copy()
    if zero_cutoff:
        stamps[stamps < 0] = 0

    stamp_sz = stamps.shape[1:3]
    x, y = stamp_centered_grid(stamp_sz)
    xx = (x*x).reshape(-1, 1)
    yy = (y*y).reshape(-1, 1)
    xy = (x*y).reshape(-1, 1)

    if weight_radius is not None:
        W = np.exp(-0.5*(xx + yy) / weight_radius**2).reshape(1, -1)
    else:
        W = np.ones(stamp_sz).reshape(1, -1)

    stamps = stamps.reshape(stamps.shape[0], -1) * W
    stamps = stamps / np.sum(stamps, axis=1, keepdims=True)

    Q_xx = stamps.dot(xx)
    Q_yy = stamps.dot(yy)
    Q_xy = stamps.dot(xy)

    return Q_xx, Q_yy, Q_xy


def ellipticities(stamps, weight_radius=None, zero_cutoff=False):

    Q_xx, Q_yy, Q_xy = moments(stamps, weight_radius, zero_cutoff)

    num = Q_xx - Q_yy + 2*(1j)*Q_xy
    epsilon_denom = Q_xx + Q_yy + 2*((Q_xx*Q_yy - Q_xy**2).astype('complex64'))**0.5
    epsilon = num / epsilon_denom
    epsilon = np.hstack((np.real(epsilon), np.imag(epsilon)))
    chi_denom = Q_xx + Q_yy
    chi = num / chi_denom
    chi = np.hstack((np.real(chi), np.imag(chi)))

    return epsilon, chi


def kron_radii(stamps):

    stamp_sz = stamps.shape[1:3]
    x, y = stamp_centered_grid(stamp_sz)
    r_sqr = (x**2 + y**2).reshape(-1,1)
    r = np.sqrt(r_sqr)
    stamps = stamps.reshape(stamps.shape[0], -1)
    kr = stamps.dot(r_sqr) / stamps.dot(r)
    return kr.flatten()


def second_moment_radii(stamps):
    
    Q_xx, Q_yy, Q_xy = moments(stamps)
    r = np.sqrt((Q_xx + Q_yy) / 2.0)
    return r.flatten()


def area_at_half_max_radii(stamps):

    stamps = norm_max(stamps)    
    num_fg_pixels = np.sum(1.0 * (stamps > 0.5), axis=(1,2))    
    r = np.sqrt(num_fg_pixels.astype('float32') / ma.pi)
    return r.flatten()


def half_light_radii(stamps):

    stamp_sz = stamps.shape[1:3]
    x, y = stamp_centered_grid(stamp_sz)
    r = np.sqrt(x**2 + y**2).flatten()
    r_argsort = np.argsort(r)
    r_sort = r[r_argsort]
    hlr = np.zeros((stamps.shape[0],))

    for i in range(stamps.shape[0]):
        curr_stamp = stamps[i, ...].flatten()
        cumsum_stamp_sort = np.cumsum(curr_stamp[r_argsort])

        if cumsum_stamp_sort[-1] > 0:
            cumsum_stamp_sort = cumsum_stamp_sort / cumsum_stamp_sort[-1]
            r_ind = np.where(cumsum_stamp_sort >= 0.5)[0][0]
            hlr[i] = r_sort[r_ind]
        else:
            hlr[i] = -1.0
            
    return hlr


def galsim_pixel_radii(stamps):

    stamp_sz = stamps.shape[1:3]
    stamps = stamps.reshape((-1, *stamp_sz))

    hlr = np.zeros((stamps.shape[0],))    
    fwhm = np.zeros((stamps.shape[0],))    
    moment = np.zeros((stamps.shape[0],))

    for i in range(stamps.shape[0]):
        curr_stamp = stamps[i, :, :]
        gsi = gsimage.Image(curr_stamp, scale=1.0)
        hlr[i] = gsi.calculateHLR()
        fwhm[i] = gsi.calculateFWHM()
        moment[i] = gsi.calculateMomentRadius()

    return hlr, fwhm, moment