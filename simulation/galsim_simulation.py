#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK

import os
import warnings
import random
import numpy as np
import galsim
import pyfits
import json
import pandas as pd

from copy import copy
from wlenet.misc.argconf import print_summary
from wlenet.misc.progress import progress
from wlenet.misc.struct import Struct
from wlenet.simulation import shear_distrib
from wlenet import config


class GalsimSimulation:

    def render_stamp(self, gal_orig, g1, g2, theta, xshift, yshift, mu, mirror):

        # transform
        gal = gal_orig.copy()
        gal.applyDilation(self.args.size_rescale)
        gal.applyRotation(theta*galsim.radians)
        gal = gal if not mirror else gal.transform(-1.0, 0.0, 0.0, 1.0)
        gal.applyLensing(g1=g1, g2=g2, mu=mu)        

        # convolve
        psf = galsim.Gaussian(fwhm=self.args.gaussian_psf_fwhm)
        pixel = galsim.Pixel(scale=self.args.pixel_scale)
        final = galsim.Convolve([psf, pixel, gal])

        # render at subpixel translation
        offset = galsim.PositionD(xshift, yshift)
        bbox = galsim.BoundsI(0, self.args.stamp_full_sz-1, 0, self.args.stamp_full_sz-1)
        galaxy_image = galsim.ImageF(self.args.stamp_full_sz, self.args.stamp_full_sz,
                                     scale=self.args.pixel_scale)
        galaxy_image.setOrigin(0, 0)
        stamp = galaxy_image.subImage(bbox)
        final.draw(stamp, normalization='flux', scale=self.args.pixel_scale, offset=offset)

        # whiten
        current_var = -1.0
        additional_var = self.args.noise_std**2
        if self.args.noise_whiten:
            current_var = final.noise.whitenImage(stamp)
            additional_var = self.args.noise_std**2 - current_var
            if additional_var < 0.0 and self.args.noise_whiten_can_fail: # pre-whitening failed
                return None, None, None, None

        # add noise
        if additional_var > 0.0:
            new_noise = galsim.GaussianNoise(self.rng, sigma=additional_var**0.5)
            new_noise.applyTo(stamp)

        return stamp, gal, current_var, additional_var

    def render_stamps(self):

        stamp_from = int((self.args.stamp_full_sz - self.args.stamp_sz) / 2)
        stamp_to = stamp_from + self.args.stamp_sz
        flux_rescale_factor =  self.args.flux_rescale_factor_extra*self.args.size_rescale**2 \
            if self.args.flux_rescale else 1.0
        noise_pad_size = int(np.ceil(self.args.stamp_full_sz * np.sqrt(2.) * self.args.pixel_scale))

        self.stamps = np.zeros((self.args.num_gals, self.args.stamp_sz, self.args.stamp_sz),
                               dtype=np.float32)
        self.shears = np.zeros((self.args.num_gals, 2), dtype=np.float32)

        metadata_columns = ['rgc_index', 'g1', 'g2', 'theta', 'xshift', 'yshift', 'mu', 'current_var', 'additional_var']
        metadata_dtypes = ['int32', *(['float32']*8)]
        self.metadata = pd.DataFrame(columns=metadata_columns)

        print('')
        whitening_retries = 0
        for i in range(self.args.num_gals // self.args.set_sz):

            if self.args.g_distrib == 'uniform':
                g1, g2 = shear_distrib.uniform(self.args.g_min, self.args.g_max)
            elif self.args.g_distrib == 'gaussian':
                g1, g2 = shear_distrib.truncated_normal(self.args.g_std, self.args.g_max)
            else:
                raise ValueError('unsupported g_distrib')

            for j in range(self.args.set_sz):

                theta = np.random.rand(1)[0] * 2 * np.pi
                mu = (np.random.rand(1)[0] - 0.5) * self.args.mu_range + 1.0
                mirror = np.random.rand(1)[0] > 0.5 if self.args.mirroring else False

                if self.args.shift_range > 0.0:
                    xshift = (np.random.rand(1)[0] - 0.5) * self.args.shift_range
                    yshift = (np.random.rand(1)[0] - 0.5) * self.args.shift_range
                else:
                    xshift = 0.0
                    yshift = 0.0

                while True:
                    rgc_index = np.random.choice(self.used_indices, size=1, p=self.used_weights)[0]

                    total_size_rescale_factor = mu * self.args.size_rescale * (self.cosmos_pixel_scale / self.args.pixel_scale)                                    
                    rescaled_flux_radius = total_size_rescale_factor * self.fit_catalog['flux_radius'][rgc_index]

                    if self.args.shift_range_proportion > 0.0: # this overrides self.args.shift_range > 0.0
                        xshift = (np.random.rand(1)[0] - 0.5) * 2.0 * rescaled_flux_radius * self.args.shift_range_proportion
                        yshift = (np.random.rand(1)[0] - 0.5) * 2.0 * rescaled_flux_radius * self.args.shift_range_proportion

                    gal_orig = galsim.RealGalaxy(self.rgc, index=rgc_index,
                                                 noise_pad_size=noise_pad_size,
                                                 flux_rescale=flux_rescale_factor)
                    stamp, gal, current_var, additional_var = self.render_stamp(gal_orig, g1, g2, theta, xshift, yshift, mu, mirror)
                    if stamp == None:
                        whitening_retries += 1

                    k = i*self.args.set_sz + j
                    progress(k+1, self.args.num_gals, status='rendered galaxy %d/%d [whitening retries %d]' %
                        (k+1, self.args.num_gals, whitening_retries))

                    if stamp != None:
                        self.shears[k, :] = [g1, g2]
                        self.stamps[k, :, :] = stamp.array[stamp_from:stamp_to, stamp_from:stamp_to]
                        self.metadata.loc[k] = [rgc_index, g1, g2, theta, xshift, yshift, mu, current_var, additional_var]
                        break
        print('\n')
        self.metadata = self.metadata.astype(dict(zip(metadata_columns, metadata_dtypes)))
        return self.stamps, self.shears, self.metadata

    def filter_valid_indices(self):

        valid_mask_orig = np.zeros((self.rgc.nobjects, 1), dtype=bool).flatten()
        valid_mask_orig[self.valid_indices] = True
        valid_mask = np.ones((self.rgc.nobjects, 1), dtype=bool).flatten()

        if self.args.mag_auto_min != -1.0:
            valid_mask = valid_mask * (self.fit_catalog['mag_auto'] > self.args.mag_auto_min)
        if self.args.mag_auto_max != -1.0:
            valid_mask = valid_mask * (self.fit_catalog['mag_auto'] < self.args.mag_auto_max)
        if self.args.flux_radius_min != -1.0:
            valid_mask = valid_mask * (self.fit_catalog['flux_radius'] > self.args.flux_radius_min)
        if self.args.flux_radius_max != -1.0:
            valid_mask = valid_mask * (self.fit_catalog['flux_radius'] < self.args.flux_radius_max)
        if self.args.zphot_min != -1.0:
            valid_mask = valid_mask * (self.fit_catalog['zphot'] > self.args.zphot_min)
        if self.args.zphot_max != -1.0:
            valid_mask = valid_mask * (self.fit_catalog['zphot'] < self.args.zphot_max)            
        if self.args.mask_dist_min != -1.0:
            valid_mask = valid_mask * (self.mask_info_catalog['MIN_MASK_DIST_FRACTION'] > self.args.mask_dist_min)
        if self.args.mask_dist_max != -1.0:
            valid_mask = valid_mask * (self.mask_info_catalog['MIN_MASK_DIST_FRACTION'] < self.args.mask_dist_max)

        valid_mask = valid_mask*valid_mask_orig
        self.valid_indices = np.where(valid_mask)[0]

    def init(self, verbose=True):

        # load the catalogs
        
        warnings.filterwarnings("ignore")
        rgc_dir, rgc_file = os.path.split(os.path.expanduser(self.args.rgc_path))
        fits_file = os.path.splitext(rgc_file)[0] + '_fits.fits'
        mask_info_file = 'real_galaxy_mask_info.fits'

        if verbose:
            print('Loading the RealGalaxyCatalog %s (preload=%d)' % (rgc_file, self.args.preload))

        self.rgc = galsim.RealGalaxyCatalog(rgc_file, dir=rgc_dir, preload=self.args.preload)
        self.rgc_catalog = pyfits.getdata(os.path.join(rgc_dir, rgc_file))
        self.fit_catalog = pyfits.getdata(os.path.join(rgc_dir, fits_file))

        if (self.args.mask_dist_min != -1.0) or (self.args.mask_dist_max != -1.0):
            self.mask_info_catalog = pyfits.getdata(os.path.join(rgc_dir, mask_info_file))            
        else:
            self.mask_info_catalog = None
        
        # possibly preload the stamps themselves
        if self.args.preload:
            for f in self.rgc.loaded_files:
                for h in self.rgc.loaded_files[f]:
                    h.data

        # possibly use python's initial seed to initialize seed_dynamic
        if self.args.seed_dynamic == -1:
            self.args.seed_dynamic = random.randint(0, 10**8 - 1)

        # possibly load valid_indices from file and then apply extra filter criteria
        no_valid_index_file = self.args.valid_index_path == ''
        self.valid_indices = np.arange(self.rgc.nobjects) if no_valid_index_file else \
                             np.load(os.path.expanduser(self.args.valid_index_path))
        if not no_valid_index_file and verbose:
            print('Using valid-index file %s' % (self.args.valid_index_path))
        self.filter_valid_indices()

        # possibly load weight_indices from file
        no_weight_index_file = self.args.weight_index_path == ''
        if not no_weight_index_file and verbose:
            print('Using weight-index file %s' % (self.args.weight_index_path))
            self.weight_indices = np.load(os.path.expanduser(self.args.weight_index_path))
        else:
            self.weight_indices = np.ones((self.rgc.nobjects,)) / self.rgc.nobjects            
        assert(self.weight_indices.shape[0] == self.rgc.nobjects)
        assert(np.all(self.weight_indices >= 0))
        assert(abs(np.sum(self.weight_indices) - 1.0) < 1e-6)

        # use seed_static to perform the train/test split
        seed_static = 12345
        if verbose:
            print('Splitting catalog samples to train and test samples using the static seed')
            print('Using the static seed for train/test split')
        np.random.seed(seed_static)
        num_indices = self.valid_indices.shape[0]
        train_test_rv = np.random.rand(self.rgc.nobjects)
        train_mask = train_test_rv < self.args.train_fraction
        test_mask = train_test_rv >= self.args.train_fraction
        valid_mask = np.zeros((self.rgc.nobjects, 1), dtype=bool).flatten()
        valid_mask[self.valid_indices] = True
        self.train_indices = np.where(train_mask * valid_mask)[0]
        self.test_indices = np.where(test_mask * valid_mask)[0]
        self.used_indices = self.train_indices if self.args.test_train == 'train' else self.test_indices

        # renormalize the weights
        self.used_weights = self.weight_indices[self.used_indices]
        self.used_weights = self.used_weights / np.sum(self.used_weights)

        # print catalog summary
        if verbose:
            print('Total catalog galaxies: %d' % (self.rgc.nobjects))
            print('Valid catalog galaxies: %d' % (self.valid_indices.shape[0]))
            print('Train catalog galaxies: %d' % (self.train_indices.shape[0]))
            print('Test catalog galaxies : %d' % (self.test_indices.shape[0]))

        # now use the dynamic seed
        seed_delta_numpy  = 50000
        seed_delta_galsim = 55000
        seed_delta_test_train = 60000
        if verbose:        
            print('Using the dynamic seed for the rest of this simulation')
        if self.args.test_train == 'test':
            np.random.seed(self.args.seed_dynamic + seed_delta_numpy)
            self.rng = galsim.UniformDeviate(self.args.seed_dynamic + seed_delta_galsim)
        else:
            np.random.seed(self.args.seed_dynamic + seed_delta_numpy + seed_delta_test_train)
            self.rng = galsim.UniformDeviate(self.args.seed_dynamic + seed_delta_galsim + seed_delta_test_train)

    def __init__(self, args=None):
        
        self.args = GalsimSimulation.default_args

        if args is None:            
            pass
        elif isinstance(args, str):
            self.args.json_config_path = args
        elif isinstance(args, dict):
            self.args.__dict__.update(args)
        elif isinstance(args, argparse.Namespace) or isinstance(args, Struct):
            self.args.__dict__.update(args.__dict__)
        else:
            raise TypeError('args should either be a string, dict, Struct or argparse.Namespace')

        if not self.args.json_config_path == '':
            with open(os.path.expanduser(self.args.json_config_path)) as json_fid:
                self.args.__dict__.update(json.load(json_fid))

        # perform some extra argument sanity checks
        assert(self.args.num_gals % self.args.set_sz == 0)

    cosmos_pixel_scale = 0.03

    default_args = Struct(**{
        'pixel_scale': 0.05,
        'size_rescale': 0.5,
        'gaussian_psf_fwhm': 0.12,
        'noise_whiten': 1,
        'noise_whiten_can_fail': 1,
        'noise_std': 0.010,
        'flux_rescale': 1,
        'flux_rescale_factor_extra': 1.0,
        'seed_dynamic': -1,
        'train_fraction': 0.85,
        'test_train': 'test',
        'num_gals': 200,
        'set_sz': 1,
        'g_min': -0.05,
        'g_max': 0.05,
        'g_std': 0.085,
        'g_distrib': 'uniform',
        'mirroring': 0,
        'shift_range': 1.0,
        'shift_range_proportion': 0.0,
        'mu_range': 0.0,
        'stamp_sz': 30,
        'stamp_full_sz': 96,
        'rgc_path': config['cosmos_path'] + '/COSMOS_23.5_training_sample/real_galaxy_catalog_23.5.fits',
        'weight_index_path': '',
        'valid_index_path': '',
        'mag_auto_min': -1.0,
        'mag_auto_max': -1.0,
        'flux_radius_min': -1.0,
        'flux_radius_max': -1.0,
        'zphot_min': -1.0,
        'zphot_max': -1.0,
        'mask_dist_min': -1.0,
        'mask_dist_max': -1.0,
        'preload': 0,
        'out_base_path': '',
        'show_stamps': 0,
        'json_config_path': ''})


if __name__ == '__main__':

    default_args = GalsimSimulation.default_args
    import argcomplete, argparse

    parser = argparse.ArgumentParser(description='Simulate sheared galaxy stamps using GalSim',
               formatter_class=lambda prog: argparse.HelpFormatter(prog,max_help_position=42, width=115))
    parser.add_argument('--pixel_scale', type=float, default=default_args.pixel_scale,
                        help='the linear dimension of a pixel in arcseconds (default: %0.2f)' %
                        (default_args.pixel_scale))
    parser.add_argument('--size_rescale', type=float, default=default_args.size_rescale,
                        help='a rescaling factor of object size (affects flux) (default: %0.2f)' %
                        (default_args.size_rescale))
    parser.add_argument('--gaussian_psf_fwhm', type=float, default=default_args.gaussian_psf_fwhm,
                        help='FWHM of simulated gaussian PSF in arcseconds (default: %0.2f)' %
                        (default_args.gaussian_psf_fwhm))
    parser.add_argument('--noise_whiten', type=int, choices=[0, 1], default=default_args.noise_whiten,
                        help='when set to 1 original stamp noise is whitened before adding extra bg noise (default: %d)' %
                        (default_args.noise_whiten))
    parser.add_argument('--noise_whiten_can_fail', type=int, choices=[0, 1], default=default_args.noise_whiten_can_fail,
                        help='when set to 1 original whitening can fail depending on noise_std (default: %d)' %
                        (default_args.noise_whiten_can_fail))    
    parser.add_argument('--noise_std', type=float, default=default_args.noise_std,
                        help='pixel noise std in units of counts/sec (default: %0.3f)' %
                        (default_args.noise_std))
    parser.add_argument('--flux_rescale', type=int, choices=[0, 1], default=default_args.flux_rescale,
                        help='when set to 1 scales flux as flux_rescale_factor_extra*size_rescale**2 (default: %d)' %
                        (default_args.flux_rescale))
    parser.add_argument('--flux_rescale_factor_extra', type=float, default=default_args.flux_rescale_factor_extra,
                        help='see flux_rescale argument (default: %d)' %
                        (default_args.flux_rescale_factor_extra))
    parser.add_argument('--seed_dynamic', type=int, default=default_args.seed_dynamic,
                        help='dynamic seed for this simulation (default: %d)' %
                        (default_args.seed_dynamic))
    parser.add_argument('--train_fraction', type=int, default=default_args.train_fraction,
                        help='fraction of COSMOS dataset samples used for training (default: %0.2f)' %
                        (default_args.train_fraction))
    parser.add_argument('--test_train', type=str, choices=['test', 'train'], default=default_args.test_train,
                        help='generate either test or train samples (default: %s)' %
                        (default_args.test_train))
    parser.add_argument('--num_gals', type=int, default=default_args.num_gals,
                        help='number of stamps to render (default: %d)' %
                        (default_args.num_gals))
    parser.add_argument('--set_sz', type=int, default=default_args.set_sz,
                        help='number of stamps in a constant shear set (default: %d)' %
                        (default_args.set_sz))
    parser.add_argument('--g_min', type=float, default=default_args.g_min,
                        help='minimal reduced shear component value to simulate (default: %0.2f)' %
                        (default_args.g_min))
    parser.add_argument('--g_max', type=float, default=default_args.g_max,
                        help='maximal reduced shear component value to simulate (default: %0.2f)' %
                        (default_args.g_max))
    parser.add_argument('--g_std', type=float, default=default_args.g_std,
                        help='when g_distrib is set to gaussian, g_std sets the standard deviation (default: %0.2f)' %
                        (default_args.g_std))
    parser.add_argument('--g_distrib', type=str, choices=['uniform', 'gaussian'], default=default_args.g_distrib,
                        help='generate g from either a uniform or a gaussian distribution (default: %s)' %
                        (default_args.g_distrib))   
    parser.add_argument('--mirroring', type=int, choices=[0, 1], default=default_args.mirroring,
                        help='allow random flipping of stamps (default: %d)' %
                        (default_args.mirroring))
    parser.add_argument('--shift_range', type=float, default=default_args.shift_range,
                        help='range of random subpixel shifts (default: %0.1f)' %
                        (default_args.shift_range))
    parser.add_argument('--shift_range_proportion', type=float, default=default_args.shift_range_proportion,
                        help='range of random subpixel shifts as proportion of scaled half-flux-radius (default: %0.1f)' %
                        (default_args.shift_range))    
    parser.add_argument('--mu_range', type=float, default=default_args.mu_range,
                        help='range of mu magnification values around 1.0 (default: %0.2f)' %
                        (default_args.mu_range))
    parser.add_argument('--stamp_sz', type=int, default=default_args.stamp_sz,
                        help='linear dimension of simulated stamps in pixels (default: %d)' %
                        (default_args.stamp_sz))
    parser.add_argument('--stamp_full_sz', type=int, default=default_args.stamp_full_sz,
                        help='linear dimension of intermediate stamps in pixels (default: %d)' %
                        (default_args.stamp_full_sz))
    parser.add_argument('--rgc_path', type=str, default=default_args.rgc_path,
                        help='path to COSMOS dataset catalog (default: %s)' %
                        (default_args.rgc_path))
    parser.add_argument('--valid_index_path', type=str, default=default_args.valid_index_path,
                        help='path to a file containing the valid rgc indices (default: %s)' %
                        (default_args.valid_index_path))
    parser.add_argument('--weight_index_path', type=str, default=default_args.weight_index_path,
                        help='path to an npy file containing the statistical weights of rgc indices (default: %s)' %
                        (default_args.weight_index_path))    
    parser.add_argument('--mag_auto_min', type=float, default=default_args.mag_auto_min,
                        help='extra input catalog filter criterion (default: %s)' %
                        (default_args.mag_auto_min))
    parser.add_argument('--mag_auto_max', type=float, default=default_args.mag_auto_max,
                        help='extra input catalog filter criterion (default: %s)' %
                        (default_args.mag_auto_max))
    parser.add_argument('--flux_radius_min', type=float, default=default_args.flux_radius_min,
                        help='extra input catalog filter criterion (default: %s)' %
                        (default_args.flux_radius_min))
    parser.add_argument('--flux_radius_max', type=float, default=default_args.flux_radius_max,
                        help='extra input catalog filter criterion (default: %s)' %
                        (default_args.flux_radius_max))
    parser.add_argument('--zphot_min', type=float, default=default_args.zphot_min,
                        help='extra input catalog filter criterion (default: %s)' %
                        (default_args.zphot_min))
    parser.add_argument('--zphot_max', type=float, default=default_args.zphot_max,
                        help='extra input catalog filter criterion (default: %s)' %
                        (default_args.zphot_max))
    parser.add_argument('--mask_dist_min', type=float, default=default_args.mask_dist_min,
                        help='extra input catalog filter criterion (default: %s)' %
                        (default_args.mask_dist_min))
    parser.add_argument('--mask_dist_max', type=float, default=default_args.mask_dist_max,
                        help='extra input catalog filter criterion (default: %s)' %
                        (default_args.mask_dist_max))    
    parser.add_argument('--preload', type=int, choices=[0, 1], default=default_args.preload,
                        help='when set to 1 stamps are loaded into memory at init (default: %d)' %
                        (default_args.preload))
    parser.add_argument('--out_base_path', type=str, default=default_args.out_base_path,
                        help='save stamps and shears to this base path (default: %s)' %
                        (default_args.out_base_path))
    parser.add_argument('--show_stamps', type=int, choices=[0, 1], default=default_args.show_stamps,
                        help='when set to 1 show resulting stamps (default: %d)' %
                        default_args.show_stamps)
    parser.add_argument('--json_config_path', type=str, default=default_args.json_config_path,
                        help='override command line arguments with those available here (default: %s)' %
                        (default_args.json_config_path))

    argcomplete.autocomplete(parser)
    args = parser.parse_args()
    gss = GalsimSimulation(args)
    print_summary(gss.args)
    gss.init()
    gss.render_stamps()

    if args.show_stamps:
        from wlenet.misc.figures import show_stamps
        print('Showing stamps')
        show_stamps(gss.stamps)

    if not args.out_base_path == '':
        print('Saving stamps to the base path: %s' % (args.out_base_path))
        gss.stamps.tofile(os.path.expanduser(args.out_base_path) + \
                          '/%s_%07d_samples.bin' % (args.test_train, args.seed_dynamic))
        gss.shears[::args.set_sz].tofile(os.path.expanduser(args.out_base_path) + \
                          '/%s_%07d_labels.bin' % (args.test_train, args.seed_dynamic))
        gss.metadata.to_pickle(os.path.expanduser(args.out_base_path) + \
                          '/%s_%07d_metadata.pkl' % (args.test_train, args.seed_dynamic), compression=None)