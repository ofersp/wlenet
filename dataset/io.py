import numpy as np
import pandas as pd

from os.path import expanduser
from json import load as load_json
from scipy.io import loadmat, savemat
from os.path import exists, isfile
from wlenet import config
from wlenet.simulation.galsim_simulation import GalsimSimulation


def load_sim(sim_names, set_name, 
             output_dim=None, stamp_sz=(32, 32), 
             norm_func=None, permute=True, post_sim=None):

    x = np.zeros((0, stamp_sz[0], stamp_sz[1], 1), dtype=np.float32)
    y = np.zeros((0, 2), dtype=np.float32)

    for sim_name in sim_names:

        sims_path = config['simulation_path']
        header_path = '%s/%s/data/%s_header.mat' % (sims_path, sim_name, set_name)
        metadata_path = '%s/%s/data/%s_metadata.pkl' % (sims_path, sim_name, set_name)
        sim_json_path = '%s/%s/configs/%s_sim.json' % (sims_path, sim_name, sim_name)        

        x_, y_ = load_lensed_stamps(expanduser(header_path))
        x_ = x_.reshape((-1,stamp_sz[0], stamp_sz[1], 1))        

        if post_sim is not None:

            # load this sim's metadata file
            metadata = pd.read_pickle(expanduser(metadata_path))
            rgc_ind = metadata['rgc_index']

            # load a gss based on sim_json_path to obtain access to the relevant rgc_catalog
            gss = GalsimSimulation(sim_json_path)
            gss.args.preload = False
            gss.init(verbose=False)

            # cutoff and factor the stamp intensities
            if 'intensity_cutoff' in post_sim:
                x_[x_ < post_sim['intensity_cutoff']] = 0
            if 'flux_factor' in post_sim:
                x_ = x_ * post_sim['flux_factor']

            # perform stamp dependent cuts
            fluxes = np.sum(x_, axis=(1, 2, 3))
            inds = np.ones(fluxes.shape).astype('bool')
            if 'flux_cut_min' in post_sim:
                inds = inds * (fluxes > post_sim['flux_cut_min'])
            if 'flux_cut_max' in post_sim:
                inds = inds * (fluxes < post_sim['flux_cut_max'])

            # perform catalog dependent cuts
            if 'mag_auto_min' in post_sim:
                inds = inds * (gss.fit_catalog['mag_auto'][rgc_ind] > post_sim['mag_auto_min'])
            if 'mag_auto_max' in post_sim:
                inds = inds * (gss.fit_catalog['mag_auto'][rgc_ind] < post_sim['mag_auto_max'])
            if 'flux_radius_min' in post_sim:
                inds = inds * (gss.fit_catalog['flux_radius'][rgc_ind] > post_sim['flux_radius_min'])
            if 'flux_radius_max' in post_sim:
                inds = inds * (gss.fit_catalog['flux_radius'][rgc_ind] < post_sim['flux_radius_max'])
            if 'zphot_min' in post_sim:
                inds = inds * (gss.fit_catalog['zphot'][rgc_ind] > post_sim['zphot_min'])
            if 'zphot_max' in post_sim:
                inds = inds * (gss.fit_catalog['zphot'][rgc_ind] < post_sim['zphot_max'])

            x_ = x_[inds, :, :, :]
            y_ = y_[inds, :]

        x = np.vstack((x, x_))
        y = np.vstack((y, y_))

    del x_; del y_

    if output_dim is not None:
        y = np.hstack((y, np.zeros((y.shape[0], output_dim - 2), dtype=np.float32)))

    if norm_func is not None:
        x = norm_func(x)

    if permute:
        p = np.random.permutation(x.shape[0])
        x = x[p, ...]
        y = y[p, ...]

    return x, y


def load_lensed_stamps(header_path):

    header_path = expanduser(header_path)
    base_path = header_path[:-10]
    samples_path = base_path + 'samples.bin'
    labels_path = base_path + 'labels.bin'
    h = loadmat(header_path)
    sz = int(h['sample_dim']**0.5)
    assert(sz**2 == h['sample_dim'])
    samples = np.fromfile(samples_path, np.float32).reshape(-1, sz, sz)
    if exists(labels_path) and isfile(labels_path):
        labels = np.fromfile(labels_path, np.float32).reshape(-1, 2)
        assert(samples.shape[0] % labels.shape[0] == 0)
        samples = samples.reshape(-1, int(samples.shape[0]/labels.shape[0]), sz, sz)
        return samples, labels
    else:
        return samples


def save_lensed_stamps(samples, labels, header_path):

    assert(len(samples) == len(labels))
    assert(len(samples.shape) >= 3)

    samples = samples.copy().astype('float32')
    labels = labels.copy().astype('float32')
    header = dict(sample_class='float32', sample_dim=np.prod(samples.shape[1:3]), sample_scale=1.0)

    header_path = expanduser(header_path)
    base_path = header_path[:-10]
    samples_path = base_path + 'samples.bin'
    labels_path = base_path + 'labels.bin'

    samples.tofile(samples_path)
    labels.tofile(labels_path)    
    savemat(header_path, header)


def load_target(header_path, stamp_sz=(32, 32), train_ratio=0.85):

    if header_path is None:
        return None, None
        
    target_stamps, _ = load_lensed_stamps(header_path)
    target_stamps = target_stamps.reshape((-1, *stamp_sz, 1))

    perm = np.random.permutation(len(target_stamps)) # TODO: manage seed

    target_stamps = target_stamps[perm, ...]
    target_n_train = int(train_ratio * len(target_stamps))
    target_n_test = len(target_stamps) - target_n_train

    target_test  = {'x': target_stamps[target_n_train:, ...]}
    target_train = {'x': target_stamps[:target_n_train, ...]}

    return target_test, target_train
