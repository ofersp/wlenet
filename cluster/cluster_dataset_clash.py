import numpy as np

from glob import glob
from os.path import expanduser
from numpy import loadtxt

from wlenet import config
from wlenet.reduction import utils as reduction_utils
from wlenet.cluster.cluster_dataset import ClusterDataset


class ClusterDatasetClash(ClusterDataset):

    def __init__(self, clash_path=None):
        self.clash_path = expanduser(config['clash_path'] if clash_path is None else clash_path)
        self.cluster_table_full = reduction_utils.load_table(expanduser(config['calibration_path'] + '/clash_clusters.txt'))
        self.cluster_ids = self.cluster_table_full['id'][self.cluster_table_full['valid'] == 1]
        self.cluster_names = self.cluster_table_full['name'][self.cluster_table_full['valid'] == 1]
        self.cluster_full_names = self.cluster_table_full['full_name'][self.cluster_table_full['valid'] == 1]
        self.cluster_redshifts = self.cluster_table_full['redshift'][self.cluster_table_full['valid'] == 1]
        self.cluster_manual_center_x = self.cluster_table_full['manual_center_x'][self.cluster_table_full['valid'] == 1]
        self.cluster_manual_center_y = self.cluster_table_full['manual_center_y'][self.cluster_table_full['valid'] == 1]
        self.cluster_names_dict = dict((n, i) for i, n in enumerate(self.cluster_names))
        self.default_catalog_name_redshift = 'hst_acs_ir'
        self.default_catalog_name_shape = 'rrg'
        self.pixel_scale = 0.065        
        self.camera_names = ['hst_acs', 'hst_acs-wfc3ir']
        self.filter_names = ['f435w', 'f475w', 'f606w', 'f625w', 'f775w', 'f814w', 'f850lp']

    def get_center_xy(self, cluster_name):        
        x = self.cluster_manual_center_x[self.cluster_names_dict[cluster_name]] * (5000 / 212.0)
        y = self.cluster_manual_center_y[self.cluster_names_dict[cluster_name]] * (5000 / 205.5); y = 5000 - y
        return x, y

    def get_pixel_scale(self):
        return self.pixel_scale

    def get_filter_names(self):
        return self.filter_names

    def get_camera_names(self):
        return self.camera_names

    def get_redshift(self, cluster_name):
        return self.cluster_redshifts[self.cluster_names_dict[cluster_name]]

    def get_id(self, cluster_name):
        return self.cluster_ids[self.cluster_names_dict[cluster_name]]

    def get_full_name(self, cluster_name):
        return self.cluster_full_names[self.cluster_names_dict[cluster_name]]

    def get_image_paths(self, cluster_name, filter_name, camera_name='hst_acs'):
      
        image_path = '%s/%s/data/hst/scale_65mas/hlsp_clash_%s_%s_%s_v1_drz.fits' % \
            (self.clash_path, cluster_name, camera_name, cluster_name, filter_name)
        image_wht_path = '%s/%s/data/hst/scale_65mas/hlsp_clash_%s_%s_%s_v1_wht.fits' % \
            (self.clash_path, cluster_name, camera_name, cluster_name, filter_name)
        return image_path, image_wht_path

    def get_redshift_catalog_specs(self, cluster_name):

        hst_acs_ir = {'name': 'hst_acs_ir',
                      'path': '%s/%s/catalogs/hst/hlsp_clash_hst_acs-ir_%s_cat.txt' % (self.clash_path, cluster_name, cluster_name),
                      'field_names': {'id': 'id', 'x': 'x', 'y': 'y', 'ra': 'RA', 'dec': 'Dec', 
                                      'z': 'zb', 'z_min': 'zbmin', 'z_max': 'zbmax', 'area': 'area', 'stel': 'stel', 'photo_flag': 'flag5sig',
                                      'ell': 'ell', 'a': None, 'b': None, 'theta': None, 
                                      'flux_radius': None, 'mag_auto': None, 'mag_iso': 'f814w_mag','sn': None}}
        hst_ir = {'name': 'hst_ir',
                  'path': '%s/%s/catalogs/hst/hlsp_clash_hst_ir_%s_cat.txt' % (self.clash_path, cluster_name, cluster_name),
                  'field_names': {'id': 'id', 'x': 'x', 'y': 'y', 'ra': 'RA', 'dec': 'Dec', 
                                  'z': 'zb', 'z_min': 'zbmin', 'z_max': 'zbmax', 'area': 'area', 'stel': 'stel', 'photo_flag': 'flag5sig',
                                  'ell': 'ell', 'a': None, 'b': None, 'theta': None, 
                                  'flux_radius': None, 'mag_auto': None, 'mag_iso': 'f814w_mag', 'sn': None}}

        hst_ir_molino = {'name': 'hst_ir_molino',
                         'path': '%s/%s/catalogs/molino/hlsp_clash_hst_ir_%s_cat-molino.txt' % (self.clash_path, cluster_name, cluster_name),
                         'field_names': {'id': 'CLASHID', 'x': 'x', 'y': 'y', 'ra': 'RA', 'dec': 'Dec', 
                                         'z': 'zb_1', 'z_min': 'zb_Min_1', 'z_max': 'zb_Max_1', 'area': 'area', 'stel': 'PointSource', 'photo_flag': 'photoflag',
                                         'ell': None, 'a': 'a', 'b': 'b', 'theta': 'theta', 
                                         'flux_radius': 'rf', 'mag_auto': None, 'mag_iso': 'F814W_ACS_MASS', 'sn': 's2n'}}
        catalog_specs = [hst_acs_ir, hst_ir, hst_ir_molino]
        return catalog_specs

    def get_shape_catalog_specs(self, cluster_name):

        path = '%s/%s/catalogs/wl/*.cat' % (self.clash_path, cluster_name)
        path = glob(path)[0]

        rrg = {'name': 'rrg',
               'path': path,
               'field_names': {'id': None, 'x': None, 'y': None, 'ra': 'RA', 'dec': 'DEC', 'g1': 'G1', 'g2': 'G2',
                               'z': 'zb', 'z_min': 'zb_min', 'z_max': 'zb_max', 'sn': 'SN'}}
        catalog_specs = [rrg]
        return catalog_specs