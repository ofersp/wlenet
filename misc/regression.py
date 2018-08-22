import numpy as np
import sympy as sp


def linreg_eiv_loglike(x, y, sxx, syy, sxy, thetas, b_perps):
    """
    Based on eqs. 26-32 of Hogg et al. - Fitting a straight line to data
    """

    thetas, b_perps = np.meshgrid(thetas, b_perps)

    orig_shape = thetas.shape
    thetas = thetas.reshape(-1, 1)
    b_perps = b_perps.reshape(-1, 1)

    v = np.column_stack((-np.sin(thetas), np.cos(thetas)))
    Z = np.column_stack((x, y))
    S = np.column_stack((sxx, syy, sxy))
    Delta = (v @ Z.T) - b_perps
    Sigma_sqr = np.column_stack((v[:,0]**2, v[:,1]**2, 2*v[:,0]*v[:,1])) @ S.T
    loglike = -0.5 * np.sum(Delta**2 / Sigma_sqr, axis=1)

    return loglike.reshape(orig_shape)


def transform_gaussian(mu_x, cov_x, x, y):
    
    eval_pt = {x[0]: mu_x[0], x[1]: mu_x[1]}
    mu_y = y.subs(eval_pt)
    J = y.jacobian(x).subs(eval_pt)
    cov_y = J * cov_x * J.T
    
    return mu_y, cov_y


def fit_gaussian_to_density_2d(rho, x, y):

    assert(np.all(rho >= 0))
    rho = rho / np.sum(rho)

    grid = np.stack(np.meshgrid(x, y), axis=0)
    mu = np.sum(grid * rho.reshape((1, *rho.shape)), axis=(1,2))
    x = grid.reshape((2, -1)) - mu.reshape((2, 1))
    rho_x = x * rho.reshape((1, -1))
    cov = (rho_x @ x.T)

    return mu, cov


def fit_gaussian_to_density_3d(rho, x, y, z):

    assert(np.all(rho >= 0))
    rho = rho / np.sum(rho)

    grid = np.mgrid[x[0]:x[1]:x[2], y[0]:y[1]:y[2], z[0]:z[1]:z[2]]
    mu = np.sum(grid * rho.reshape((1, *rho.shape)), axis=(1,2,3))
    x = grid.reshape((3, -1)) - mu.reshape((3, 1))
    rho_x = x * rho.reshape((1, -1))
    cov = (rho_x @ x.T)

    return mu, cov
