import numpy as np
from keras import backend as K


def gaussian_nll(y_true, y_pred):

    nll = gaussian_nll_per_sample(y_true, y_pred)
    mean_nll = K.mean(nll, axis=0)    
    return mean_nll


def gaussian_nll_weighted(y_true, y_pred):

    weights = y_true[:, 2]
    nll = gaussian_nll_per_sample(y_true, y_pred)
    mean_nll_weighted = K.mean(nll * weights, axis=0)
    return mean_nll_weighted


def gaussian_nll_per_sample(y_true, y_pred):

    inv_sigma_11, inv_sigma_22, inv_sigma_12 = gaussian_inv_sigmas(y_pred)

    log_det_inv_sigma = K.log(inv_sigma_11*inv_sigma_22 - inv_sigma_12**2)
    
    delta_1 = y_true[:, 0] - y_pred[:, 0]
    delta_2 = y_true[:, 1] - y_pred[:, 1]
    
    nll_a = -log_det_inv_sigma    
    nll_b = inv_sigma_11*delta_1**2 + inv_sigma_22*delta_2**2 + 2*inv_sigma_12*delta_1*delta_2
    nll = nll_a + nll_b

    return nll


def gaussian_pred(y_pred):

    means = y_pred[:, :2]
    inv_covs = np.zeros((y_pred.shape[0], 2, 2))
    inv_sigma_11, inv_sigma_22, inv_sigma_12 = gaussian_inv_sigmas(y_pred)

    inv_covs[:, 0, 0] = inv_sigma_11
    inv_covs[:, 1, 1] = inv_sigma_22
    inv_covs[:, 0, 1] = inv_sigma_12
    inv_covs[:, 1, 0] = inv_sigma_12

    return inv_covs, means


def gaussian_inv_sigmas(y_pred):

    epsilon = 1e-4

    a11 = y_pred[:, 2]
    a12 = y_pred[:, 3]
    a21 = y_pred[:, 4]
    a22 = y_pred[:, 5]
    
    inv_sigma_11 = epsilon + a11**2 + a12**2
    inv_sigma_22 = epsilon + a21**2 + a22**2    
    inv_sigma_12 = a21*a11 + a12*a22

    return inv_sigma_11, inv_sigma_22, inv_sigma_12