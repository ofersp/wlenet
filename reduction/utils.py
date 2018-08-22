import os
import numpy as np
from scipy.signal import convolve2d
from astropy.io import ascii
from astropy.io import fits
from astropy import wcs as astro_wcs


def load_image_and_header(fits_path, hdu_index=0):

    f = fits.open(os.path.expanduser(fits_path))
    image = f[hdu_index].data
    header = f[hdu_index].header
    f.close()
    return image, header


def load_header(fits_path, hdu_index=0):

    f = fits.open(os.path.expanduser(fits_path))
    header = f[hdu_index].header
    f.close()
    return header


def load_image_and_wcs(fits_path, hdu_index=0):

    f = fits.open(os.path.expanduser(fits_path))
    image = f[hdu_index].data
    wcs = astro_wcs.WCS(f[hdu_index].header)
    f.close()
    return image, wcs


def load_wcs(fits_path, hdu_index=0):

    f = fits.open(os.path.expanduser(fits_path))
    wcs = astro_wcs.WCS(f[hdu_index].header)
    f.close()
    return wcs


def load_table(table_path):

    t = ascii.read(os.path.expanduser(table_path))
    return t


def load_column(table_path, column_name):

    t = load_table(table_path)
    if column_name in t.colnames:
        return t[column_name]
    else:
        return None

    
def get_ra_dec(table, ra_name, dec_name, ra_in_hours=False):

    ra_factor = 15.0 if ra_in_hours else 1.0
    dec_factor = 1.0   
    ra = table[ra_name] * ra_factor
    dec = table[dec_name] * dec_factor
    ra_dec = np.vstack((ra, dec)).T
    return ra_dec


def load_ra_dec(table_path, ra_name, dec_name, ra_in_hours=False):

    table = load_table(table_path)
    ra_dec = get_ra_dec(table, ra_name, dec_name, ra_in_hours)
    return ra_dec


def ra_dec_to_pix(ra_dec, wcs, delta=(0, 0)):

    delta = np.array(delta)
    ra_dec_pix = wcs.wcs_world2pix(ra_dec, 0)
    ra_dec_pix = ra_dec_pix + delta
    return ra_dec_pix
