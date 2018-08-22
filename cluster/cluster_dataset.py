import numpy as np
from os.path import expanduser

from astropy.io import fits
from wlenet.reduction import utils as reduction_utils
from wlenet import config
from wlenet.misc.cosmology import AngularDiameterDistance

class ClusterDataset(object):

    def __str__(self):
        raise NotImplementedError()

    def get_center_xy(self, cluster_name):
        raise NotImplementedError()

    def get_pixel_scale(self):
        raise NotImplementedError()

    def get_filter_names(self):
        raise NotImplementedError()

    def get_camera_names(self):
        raise NotImplementedError()

    def get_id(self, cluster_name):
        raise NotImplementedError()

    def get_full_name(self, cluster_name):
        raise NotImplementedError()

    def get_redshift(self, cluster_name):
        raise NotImplementedError()

    def get_arcsec_to_kpc(self, cluster_name):
        z = self.get_redshift(cluster_name)
        add = AngularDiameterDistance()
        d_ang = add.evaluate_0(z) # angular diameter distance in Mpc/rad
        arcsec_to_kpc = (1000.0 / 206265) * d_ang
        return arcsec_to_kpc

    def get_image_paths(self, cluster_name, filter_name, camera_name='hst_acs'):
        raise NotImplementedError()

    def get_redshift_catalog_specs(self, cluster_name):
        raise NotImplementedError()

    def get_shape_catalog_specs(self, cluster_name):
        raise NotImplementedError()

    def load_images_and_wcs(self, cluster_name, filter_name, camera_name='hst_acs'):
        image_path, image_wht_path = self.get_image_paths(cluster_name, filter_name, camera_name)
        image, wcs = reduction_utils.load_image_and_wcs(image_path, 0)        
        image_wht = reduction_utils.load_image_and_wcs(image_wht_path, 0)[0]
        return image, image_wht, wcs

    def load_redshift_catalogs(self, cluster_name, wcs=None, wcs_force_use=False):

        specs = self.get_redshift_catalog_specs(cluster_name)
        catalogs = {}
        catalogs_orig = {}

        for i, spec in enumerate(specs):

            cat = {}
            orig = reduction_utils.load_table(spec['path'])
            fields = spec['field_names']

            for k in fields:
                if fields[k] is not None:
                    cat[k] = orig[fields[k]]

            if fields['ra'] is not None and fields['dec'] is not None:
                cat['radec'] = np.vstack((orig[fields['ra']], orig[fields['dec']])).T
            if fields['x'] is not None and fields['y'] is not None:
                cat['xy'] = np.vstack((orig[fields['x']], orig[fields['y']])).T - 1.0
            if (fields['x'] is None or fields['y'] is None or wcs_force_use) and ('radec' in cat and wcs is not None):
                cat['xy'] = reduction_utils.ra_dec_to_pix(cat['radec'], wcs)            
            if fields['ell'] is None and (fields['a'] is not None and fields['b'] is not None):
                cat['ell'] = 1.0 - orig[fields['b']] / orig[fields['a']]

            assert('id' in cat and 'xy' in cat and \
                   'z' in cat and 'z_min' in cat and 'z_max' in cat and \
                   'stel' in cat and 'ell' in cat)

            catalogs[spec['name']] = cat
            catalogs_orig[spec['name']] = orig

        assert(len(catalogs) > 0)
        return catalogs, catalogs_orig

    def load_shape_catalogs(self, cluster_name, wcs=None, wcs_force_use=False):

        specs = self.get_shape_catalog_specs(cluster_name)
        catalogs = {}
        catalogs_orig = {}

        for i, spec in enumerate(specs):

            cat = {}
            orig = reduction_utils.load_table(spec['path'])
            fields = spec['field_names']

            for k in fields:
                if fields[k] is not None:
                    cat[k] = orig[fields[k]]

            if fields['g1'] is not None and fields['g2'] is not None:
                cat['shapes'] = np.vstack((orig[fields['g1']], orig[fields['g2']])).T
            if fields['ra'] is not None and fields['dec'] is not None:
                cat['radec'] = np.vstack((orig[fields['ra']], orig[fields['dec']])).T
            if fields['x'] is not None and fields['y'] is not None:
                cat['xy'] = np.vstack((orig[fields['x']], orig[fields['y']])).T
            if (fields['x'] is None or fields['y'] is None or wcs_force_use) and ('radec' in cat and wcs is not None):
                cat['xy'] = reduction_utils.ra_dec_to_pix(cat['radec'], wcs)

            assert('xy' in cat and 'z' in cat and 'z_min' in cat and 'z_max' in cat and 'shapes' in cat and 'sn' in cat)

            catalogs[spec['name']] = cat
            catalogs_orig[spec['name']] = orig

        assert(len(catalogs) > 0)
        return catalogs, catalogs_orig

    def compute_sextractor_catalog(self, image_path, image_wht_path,
                                   detect_minarea=9, phot_fluxfrac=0.5, 
                                   detect_thresh=1.0, analysis_thresh=1.0, 
                                   deblend_nthresh=32, deblend_mincont=0.0015,
                                   pixel_scale=0.065, weight_thresh=100, 
                                   save_check_images=False, use_image_wht=True):
        import sewpy

        sexpath = expanduser(config['sextractor_path'])
        params=["X_IMAGE", "Y_IMAGE", "FLUX_RADIUS", "FLUX_AUTO", "MAG_AUTO", "MU_MAX", "FWHM_IMAGE", "FLAGS"]

        config_se={"DETECT_MINAREA":detect_minarea, "PHOT_FLUXFRAC": phot_fluxfrac, 
                   "DETECT_THRESH": detect_thresh, "ANALYSIS_THRESH": analysis_thresh,
                   "DEBLEND_NTHRESH": deblend_nthresh, "DEBLEND_MINCONT": deblend_mincont,
                   "PIXEL_SCALE": pixel_scale}

        if save_check_images:
            config_se.update({"CHECKIMAGE_TYPE": "BACKGROUND,BACKGROUND_RMS,OBJECTS,-BACKGROUND,-OBJECTS,APERTURES,SEGMENTATION"})
        if use_image_wht:
            config_se.update({"WEIGHT_IMAGE": image_wht_path, "WEIGHT_TYPE": "MAP_WEIGHT", "WEIGHT_THRESH": weight_thresh})

        sew = sewpy.SEW(sexpath=sexpath, params=params, config=config_se)
        catalog_orig = sew(image_path)

        catalog = {}
        catalog['xy'] = np.vstack((catalog_orig['table']['X_IMAGE'], catalog_orig['table']['Y_IMAGE'])).T - 1.0
        catalog['flux_radius'] = catalog_orig['table']['FLUX_RADIUS']
        catalog['flux_auto'] = catalog_orig['table']['FLUX_AUTO']
        catalog['mag_auto'] = catalog_orig['table']['MAG_AUTO']
        catalog['mu_max'] = catalog_orig['table']['MU_MAX']
        catalog['fwhm_image'] = catalog_orig['table']['FWHM_IMAGE']

        return catalog, catalog_orig

    def match_catalogs(self, cat1, cat2, min_deltas={'xy': 10, 'z': 0.1}, two_way=False, match_z=False):

        if two_way or match_z:
            raise NotImplementedError()

        inds = np.zeros((cat1['xy'].shape[0],), dtype=np.int32)
        valid = np.zeros((cat1['xy'].shape[0],), dtype=np.bool)

        for i, radec in enumerate(cat1['xy']):
            deltas = np.sum((cat1['xy'][[i], :] - cat2['xy'])**2, axis=1)
            inds[i] = np.argmin(deltas)
            valid[i] = deltas[inds[i]] < min_deltas['xy']**2

        matches = {'inds': inds, 'valid': valid}
        return matches

    def compute_total_images(self, cluster_name, camera_name='hst_acs', variance_weighted=True, wht_thresh_bad_pix=1.0, save_fits=False):

        filter_names = self.get_filter_names()

        if variance_weighted:
            total_image, total_image_wht, total_wcs = self.load_images_and_wcs(cluster_name, filter_names[0], camera_name)
            total_image *= total_image_wht

            for i in range(1, len(filter_names)):
                filter_image, filter_image_wht, filter_wcs = self.load_images_and_wcs(cluster_name, filter_names[i], camera_name)
                total_image += filter_image * filter_image_wht
                total_image_wht += filter_image_wht

            total_bad_pix = total_image_wht < wht_thresh_bad_pix
            total_image[~total_bad_pix] /= total_image_wht[~total_bad_pix]

        else:
            total_image, total_image_wht, total_wcs = self.load_images_and_wcs(cluster_name, filter_names[0], camera_name)
            total_bad_pix = total_image_wht < wht_thresh_bad_pix
            total_image_wht[total_bad_pix] = 1.0
            total_image_var = total_image_wht**-1

            for i in range(1, len(filter_names)):
                filter_image, filter_image_wht, filter_wcs = self.load_images_and_wcs(cluster_name, filter_names[i], camera_name)
                filter_bad_pix = filter_image_wht < wht_thresh_bad_pix
                total_image += filter_image
                filter_image_wht[filter_bad_pix] = 1.0
                total_image_var += filter_image_wht**-1
                total_bad_pix |= filter_bad_pix

            total_image_wht = total_image_var**-1
            total_image_wht *= len(filter_names)**2
            total_image /= len(filter_names)

        total_image[total_bad_pix] = 0
        total_image_wht[total_bad_pix] = 0

        if not save_fits:
            return total_image, total_image_wht, total_bad_pix, total_wcs

        total_inp_path, total_wht_inp_path = self.get_image_paths(cluster_name, filter_name='total', camera_name=camera_name)
        total_out_path = '/tmp/computed_total_%s_%s_%s_drz.fits' % ('weighted' if variance_weighted else 'unweighted', camera_name, cluster_name)
        total_wht_out_path = '/tmp/computed_total_%s_%s_%s_drw.fits' % ('weighted' if variance_weighted else 'unweighted', camera_name, cluster_name)

        total_header = reduction_utils.load_header(total_inp_path)
        total_wht_header = reduction_utils.load_header(total_wht_inp_path)

        fits.writeto(total_out_path, total_image, total_header, overwrite=True)
        fits.writeto(total_wht_out_path, total_image_wht, total_wht_header, overwrite=True)

        return total_image, total_image_wht, total_bad_pix, total_wcs, total_out_path, total_wht_out_path