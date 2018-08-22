import numpy as np

from copy import copy
from itertools import product
from multiprocessing import Pool, cpu_count
from astropy.table import Table

from wlenet.cluster.cluster_lensing import ClusterLensing


class ClusterSurvey(object):

    def __init__(self, cluster_template=None, mp_pool_size=None):

        self.mp_pool_size = cpu_count() - 1 if mp_pool_size is None else mp_pool_size
        self.cluster_template = cluster_template if cluster_template is not None else ClusterLensing()
        self.cluster_dataset = self.cluster_template.cluster_dataset
        self.cluster_names = []
        self.clusters = []
        self.loaded = False
        self.processed = False

    def process_survey(self, cluster_names=None, parallel=True):
    
        self.load_clusters(cluster_names, parallel)
        self.process_clusters(parallel)
        self.append_clusters()

    def load_clusters(self, cluster_names=None, parallel=True):

        cluster_names = cluster_names if cluster_names is not None else self.cluster_dataset.cluster_names
        self.cluster_names = cluster_names
        if parallel:
            pool = Pool(self.mp_pool_size)
            self.clusters = pool.starmap(self.load_cluster, product(cluster_names, [self.cluster_template]))
        else:
            self.clusters = [self.load_cluster(n, self.cluster_template) for n in cluster_names]
        self.loaded = True

    def process_clusters(self, parallel=True):

        if parallel:
            pool = Pool(self.mp_pool_size)
            self.clusters = pool.map(self.process_cluster, self.clusters)
        else:
            self.clusters = [self.process_cluster(c) for c in self.clusters]
        self.processed = True

    def append_clusters(self):

        self.cut = self.clusters[0].cut.copy()
        self.full = self.clusters[0].full.copy()

        for i in range(1, len(self.clusters)):
            for k in self.cut.keys():    
                self.cut[k] = np.append(self.cut[k], self.clusters[i].cut[k], axis=0)
            for k in self.full.keys():    
                self.full[k] = np.append(self.full[k], self.clusters[i].full[k], axis=0)               

    def remove_tmp_files(self):
        
        for cluster in self.clusters:
            cluster.remove_tmp_files()

    @staticmethod
    def load_cluster(cluster_name, cluster_template):

        print('Loading cluster: ' + cluster_name)
        cluster = copy(cluster_template)
        cluster.load(cluster_name)
        return cluster

    @staticmethod
    def process_cluster(cluster):

        print('Processing cluster: ' + cluster.name)
        cluster.process_stamps()
        cluster.process_fields()
        return cluster

    def table(self):

        mini_pixel_area = self.cluster_template.cluster_dataset.get_pixel_scale() / self.cluster_template.mini_scale_factor

        id_ = [c.id for c in self.clusters]
        name = [c.name for c in self.clusters]
        full_name = [c.full_name for c in self.clusters]
        redshift = [c.redshift for c in self.clusters]
        arcsec_to_kpc = [c.arcsec_to_kpc for c in self.clusters]

        area = [np.sum(c.image_wht_mini > c.cut_params['mean_wht_min']) * mini_pixel_area for c in self.clusters]
        num_full = [len(c.full['source_id']) for c in self.clusters]
        num_cut = [len(c.cut['source_id']) for c in self.clusters]
        num_match_phot = [np.sum(c.matches_shape_to_phot['valid']) for c in self.clusters]
        num_match_redshift = [np.sum(c.matches_shape_to_redshift['valid']) for c in self.clusters]
        mean_z = [np.mean(c.cut['z']) for c in self.clusters]
        mean_flux_radius = [np.mean(c.cut['flux_radius']) for c in self.clusters]
        mean_mag_auto = [np.mean(c.cut['mag_auto']) for c in self.clusters]
        mean_effic = [np.mean(c.cut['effic']) for c in self.clusters]
        density_cut = np.array(num_cut) / np.array(area)

        rows = [id_, name, full_name, redshift, area, num_full, num_cut, num_match_phot, num_match_redshift, 
                mean_z, mean_flux_radius, mean_mag_auto, mean_effic, density_cut]
        names = ['id', 'name', 'full_name', 'redshift', 'area', 'num_full', 'num_cut', 'num_match_phot', 'num_match_redshift', 
                 'mean_z', 'mean_flux_radius', 'mean_mag_auto', 'mean_effic', 'density_cut']

        return Table(rows, names=names)