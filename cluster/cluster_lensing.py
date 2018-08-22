import numpy as np

from shutil import rmtree
from os import remove
from os.path import expanduser, isdir, isfile
from skimage.transform import downscale_local_mean

from wlenet import config
from wlenet.cluster.utils import lensing_efficiency
from wlenet.reduction.stamps import extract
from wlenet.cluster.cluster_dataset_clash import ClusterDatasetClash


default_cut_params = {'num_bad_pix_min': -np.inf, 'num_bad_pix_max': 1,
                      'mean_wht_min': 50000, 'mean_wht_max': np.inf,
                      'flux_radius_min': 0, 'flux_radius_max': np.inf, 
                      'mag_auto_min': -np.inf, 'mag_auto_max': 98,
                      'z_min': -np.inf, 'z_max': np.inf,
                      'z_rel_err_min': -np.inf, 'z_rel_err_max': np.inf, 
                      'apply_extra_cut': False}

class ClusterLensing(object):

    def __init__(self, 
                 cluster_dataset=ClusterDatasetClash(),
                 stamp_shear_estimators={},
                 field_shear_estimators={},
                 cut_params=None,
                 stamp_sz=[32, 32],
                 use_stsci_total_images=False,
                 wht_thresh_bad_pix=100.0,
                 variance_weighted=False,
                 mini_scale_factor=0.1,
                 save_check_images=False,
                 keep_image_bad_pix=False,
                 keep_image_wht=False,
                 keep_image=False):

        self.cluster_dataset = cluster_dataset
        self.stamp_shear_estimators = stamp_shear_estimators
        self.field_shear_estimators = field_shear_estimators

        self.cut_params = default_cut_params.copy()
        if cut_params is not None:
            self.cut_params.update(cut_params)

        self.stamp_sz = np.array(stamp_sz)
        self.use_stsci_total_images = use_stsci_total_images
        self.wht_thresh_bad_pix = wht_thresh_bad_pix
        self.variance_weighted = variance_weighted
        self.mini_scale_factor = mini_scale_factor
        self.save_check_images = save_check_images
        self.keep_image_bad_pix = keep_image_bad_pix
        self.keep_image_wht = keep_image_wht
        self.keep_image = keep_image

        self.loaded = False
        self.processed_stamps = False
        self.processed_fields = False

    def load(self, cluster_name):

        self.name = cluster_name
        self.id = self.cluster_dataset.get_id(cluster_name)
        self.full_name = self.cluster_dataset.get_full_name(cluster_name)
        self.redshift = self.cluster_dataset.get_redshift(cluster_name)
        self.center_xy = self.cluster_dataset.get_center_xy(cluster_name)
        self.arcsec_to_kpc = self.cluster_dataset.get_arcsec_to_kpc(cluster_name)

        if self.use_stsci_total_images:
            self.image, self.image_wht, self.wcs = self.cluster_dataset.load_images_and_wcs(cluster_name, filter_name='total')
            self.image_fits_path, self.image_wht_fits_path = self.cluster_dataset.get_image_paths(cluster_name, filter_name='total')
            self.image_bad_pix = self.image_wht < self.wht_thresh_bad_pix
            self.image[self.image_bad_pix] = 0
            self.image_wht[self.image_bad_pix] = 0
        else:
            self.image, self.image_wht, self.image_bad_pix, self.wcs, self.image_fits_path, self.image_wht_fits_path = \
                self.cluster_dataset.compute_total_images(cluster_name, camera_name='hst_acs', 
                    variance_weighted=self.variance_weighted, wht_thresh_bad_pix=self.wht_thresh_bad_pix, save_fits=True)

        self.image_shape = self.image.shape
        
        self.image_mini = downscale_local_mean(self.image, (int(1/self.mini_scale_factor), int(1/self.mini_scale_factor)))
        self.image_wht_mini = downscale_local_mean(self.image_wht, (int(1/self.mini_scale_factor), int(1/self.mini_scale_factor)))
        self.image_bad_pix_mini = downscale_local_mean(self.image_bad_pix, (int(1/self.mini_scale_factor), int(1/self.mini_scale_factor)))

        self.catalogs_redshift, self.catalogs_redshift_orig = self.cluster_dataset.load_redshift_catalogs(self.name, self.wcs)
        self.catalogs_shape, self.catalogs_shape_orig = self.cluster_dataset.load_shape_catalogs(self.name, self.wcs)

        self.catalog_redshift = self.catalogs_redshift[self.cluster_dataset.default_catalog_name_redshift]
        self.catalog_shape = self.catalogs_shape[self.cluster_dataset.default_catalog_name_shape]

        self.catalog_phot, self.catalog_phot_orig = \
            self.cluster_dataset.compute_sextractor_catalog(
                self.image_fits_path, self.image_wht_fits_path, save_check_images=self.save_check_images, 
                weight_thresh=self.wht_thresh_bad_pix)

        self.matches_shape_to_redshift = self.cluster_dataset.match_catalogs(self.catalog_shape, self.catalog_redshift)
        self.matches_shape_to_phot = self.cluster_dataset.match_catalogs(self.catalog_shape, self.catalog_phot)
        self.loaded = True

    def process_stamps(self):

        self.extract_stamps_full()
        self.process_full()
        self.process_cut()
        self.measure_cut()
        self.processed_stamps = True

    def extract_stamps_full(self):

        stamps, status = extract(self.image, None, self.catalog_shape['xy'], self.stamp_sz)
        stamps_bad_pix, _ = extract(self.image_bad_pix, None, self.catalog_shape['xy'], self.stamp_sz)
        stamps_wht, _ = extract(self.image_wht, None, self.catalog_shape['xy'], self.stamp_sz)

        if not self.keep_image:
            self.image = None
        if not self.keep_image_wht:
            self.image_wht = None
        if not self.keep_image_bad_pix:
            self.image_bad_pix = None

        self.full = {}
        self.full['cluster_id'] = np.array([self.id] * stamps.shape[0])
        self.full['source_id'] = np.arange(0, stamps.shape[0])
        self.full['stamps'] = stamps
        self.full['extracted'] = (status.flatten() == 1)
        self.full['num_bad_pix'] = np.sum(stamps_bad_pix, axis=(1, 2))
        self.full['mean_wht'] = np.mean(stamps_wht, axis=(1, 2))

    def process_full(self):

        self.full['xy'] = self.catalog_shape['xy']
        self.full['shears_cat'] = self.catalog_shape['shapes']
        self.full['z'] = self.catalog_shape['z']
        self.full['z_min'] = self.catalog_shape['z_min']
        self.full['z_max'] = self.catalog_shape['z_max']
        self.full['z_rel_err'] = (self.full['z_max'] - self.full['z_min']) / self.full['z']
        self.full['flux_radius'] = self.catalog_phot['flux_radius'][self.matches_shape_to_phot['inds']]
        self.full['flux_auto'] = self.catalog_phot['flux_auto'][self.matches_shape_to_phot['inds']]
        self.full['mag_auto'] = self.catalog_phot['mag_auto'][self.matches_shape_to_phot['inds']]
        self.full['mag_iso'] = self.catalog_redshift['mag_iso'][self.matches_shape_to_redshift['inds']]
        self.full['stel'] = self.catalog_redshift['stel'][self.matches_shape_to_redshift['inds']]
        self.full['mu_max'] = self.catalog_phot['mu_max'][self.matches_shape_to_phot['inds']]
        self.full['fwhm_image'] = self.catalog_phot['fwhm_image'][self.matches_shape_to_phot['inds']]
        self.full['effic'] = lensing_efficiency(self.redshift, self.full['z'])

    def process_cut(self):

        self.cut_masks = {} # cut_masks are True where full should be included in cut
        self.cut_masks['extracted'] = self.full['extracted']
        self.cut_masks['matched_to_redshift'] = self.matches_shape_to_redshift['valid']
        self.cut_masks['matched_to_phot'] = self.matches_shape_to_phot['valid']
        self.cut_masks['num_bad_pix'] = (self.full['num_bad_pix'] > self.cut_params['num_bad_pix_min']) * \
                                        (self.full['num_bad_pix'] < self.cut_params['num_bad_pix_max'])
        self.cut_masks['mean_wht'] = (self.full['mean_wht'] > self.cut_params['mean_wht_min']) * \
                                     (self.full['mean_wht'] < self.cut_params['mean_wht_max'])
        self.cut_masks['flux_radius'] = (self.full['flux_radius'] > self.cut_params['flux_radius_min']) * \
                                        (self.full['flux_radius'] < self.cut_params['flux_radius_max'])
        self.cut_masks['mag_auto'] = (self.full['mag_auto'] > self.cut_params['mag_auto_min']) * \
                                     (self.full['mag_auto'] < self.cut_params['mag_auto_max'])
        self.cut_masks['z'] = (self.full['z'] > self.cut_params['z_min']) * \
                              (self.full['z'] < self.cut_params['z_max'])
        self.cut_masks['z_rel_err'] = (self.full['z_rel_err'] > self.cut_params['z_rel_err_min']) * \
                                      (self.full['z_rel_err'] < self.cut_params['z_rel_err_max'])
        self.cut_masks['final'] = np.prod([m for m in self.cut_masks.values()], axis=0) == 1

        if self.cut_params['apply_extra_cut']:
            extra_cut_path = expanduser(config['calibration_path'] + '/clash_extra_cut.npy')
            extra_cut = np.load(extra_cut_path).item()
            extra_cut_this_cluster = extra_cut['cut_mask'][extra_cut['cluster_id'] == self.id]
            assert(len(extra_cut_this_cluster) == np.sum(self.cut_masks['final']))
            final_inds = np.where(self.cut_masks['final'])[0]
            self.cut_masks['final'][final_inds] *= extra_cut_this_cluster

        self.cut = dict([(k, v[self.cut_masks['final'], ...]) for k, v in self.full.items()])

    def measure_cut(self):
        
        for sse_name in self.stamp_shear_estimators:
            sse = self.stamp_shear_estimators[sse_name]
            self.cut['shears_' + sse_name] = sse.estimate(self.cut['stamps'])
            
    def process_fields(self):

        self.shear_fields = {}
        for sfe_name in self.field_shear_estimators:
            sfe = self.field_shear_estimators[sfe_name]
            self.shear_fields[sfe_name] = sfe.estimate(self)
        self.processed_fields = True

    def remove_tmp_files(self):

        if not self.loaded:
            return

        assert('workdir' in self.catalog_phot_orig)
        sewpy_workdir = self.catalog_phot_orig['workdir'] 

        if isdir(sewpy_workdir):
            rmtree(sewpy_workdir)
        if isfile(self.image_fits_path) and not self.use_stsci_total_images:
            assert(self.image_fits_path.startswith('/tmp'))
            remove(self.image_fits_path)
        if isfile(self.image_wht_fits_path) and not self.use_stsci_total_images:
            assert(self.image_wht_fits_path.startswith('/tmp'))
            remove(self.image_wht_fits_path)