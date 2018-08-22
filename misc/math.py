import numpy as np


def gaussian_mult(inv_covs, means):

    n = inv_covs.shape[0]
    d = inv_covs.shape[1]

    assert(inv_covs.shape[1] == d)
    assert(means.shape[1] == d)
    assert(means.shape[0] == n)

    inv_cov = np.sum(inv_covs, axis=0)
    cov = np.linalg.inv(inv_cov)
    m = np.sum(inv_covs*means.reshape((n, d, 1)), axis=(0, 2)).reshape(d, 1)
    mean = cov.dot(m)

    return inv_cov, mean, cov


def gaussian_transform(inv_covs, means, A, delta):

    n = inv_covs.shape[0]
    d = inv_covs.shape[1]

    assert(inv_covs.shape[1] == d)
    assert(means.shape[1] == d)
    assert(means.shape[0] == n)

    means_tag = (A.dot(means.T)).T + delta
    A_inv = np.linalg.inv(A)
    inv_covs_tag = np.zeros(inv_covs.shape)

    for i in range(n):
        inv_covs_tag[i, :, :] = A_inv.T.dot(inv_covs[i, :, :].dot(A_inv))

    return inv_covs_tag, means_tag
