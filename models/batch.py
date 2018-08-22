import math as ma
import numpy as np
import matplotlib.pyplot as plt
from os.path import expanduser

from wlenet import config
from wlenet.models.calibrate import step_bias_calib, step_bias_correct
from wlenet.models.utils import get_output_dim, load_model, load_spec, save_spec, print_spec
from wlenet.models.train import train_model
from wlenet.dataset.io import load_sim, load_target
from wlenet.dataset.sequences import CorrelatedNoiseSequence, generate_batches
from wlenet.dataset.normalization import norm_mean_std
from wlenet.models.predict import predict_test_time_aug


def train_a_spec(model_spec, do_train=True, save_calib=True, show_figures=False, dry_run=False, fine_tune=False):

    if show_figures:
        from wlenet.dataset.figures import show_stamps
        from wlenet.models.figures import show_scatter_label_pred, show_first_conv_kernels

    print_spec(model_spec)

    model = load_model(model_spec, load_weights=False, show_summary=True)

    noise_kernel_path = config['calibration_path'] + '/' +  model_spec['kwargs_dataset']['noise_kernel_name'] + '_noise_kernel.npy'
    noise_kernel = np.load(expanduser(noise_kernel_path)) *  model_spec['kwargs_dataset']['noise_kernel_factor']
    post_sim = model_spec['kwargs_dataset']['post_sim'] if 'post_sim' in model_spec['kwargs_dataset'] else None
    aug_noise_factor = model_spec['kwargs_dataset']['aug_noise_factor'] if 'aug_noise_factor' in model_spec['kwargs_dataset'] else False
    output_dim = get_output_dim(model)
    source_test = dict(zip(('x', 'y'), load_sim(model_spec['kwargs_dataset']['sim_names_test'], 'test', output_dim, post_sim=post_sim)))
    source_train = dict(zip(('x', 'y'), load_sim(model_spec['kwargs_dataset']['sim_names_train'], 'train', output_dim, post_sim=post_sim))) if do_train else None
    target_test, target_train = load_target(model_spec['kwargs_dataset']['target_header_path'])
    seq_test = CorrelatedNoiseSequence(source_test, noise_kernel, target=target_test, aug_noise_factor=aug_noise_factor)
    seq_train = CorrelatedNoiseSequence(source_train, noise_kernel, target=target_train, aug_noise_factor=aug_noise_factor) if do_train else None

    if show_figures:
        show_stamps(generate_batches(seq_test, 1)[0], clim=[-2, 2])

    if do_train:
        model = train_model(model_spec, seq_train, seq_test, **model_spec['kwargs_train'], dry_run=dry_run, fine_tune=fine_tune)

    if show_figures:
        show_first_conv_kernels(model)

    seq_test = CorrelatedNoiseSequence(source_test, noise_kernel)
    x_seq_test, y_seq_test, inds_seq_test = generate_batches(seq_test, 500)
    y_seq_ptta = predict_test_time_aug(x_seq_test, model)
    y_seq_ptta, calib_ptta, rmse_ptta = step_bias_calib(y_seq_test, y_seq_ptta, verbose=True)

    if show_figures:
        show_scatter_label_pred(y_seq_test, y_seq_ptta, min_g=-0.6, max_g=0.6, min_g_true=-0.2, max_g_true=0.2)

    model_spec['calib'] = {'ptta': calib_ptta}
    if save_calib and not dry_run:
        save_spec(model_spec)

    return model_spec, model