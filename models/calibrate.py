import numpy as np


def step_bias_correct(g_obs, calib):
    
    c1, m1, c2, m2 = calib
    g_corr = g_obs.copy()
    g_corr[:, 0] = (g_obs[:, 0] - c1) / (1 + m1)
    g_corr[:, 1] = (g_obs[:, 1] - c2) / (1 + m2)
    return g_corr


def step_bias_correct_tx(calib):

    c1, m1, c2, m2 = calib

    A = np.zeros((2, 2))
    delta = np.zeros((1, 2))

    A[0, 0] = 1 / (1 + m1)
    A[1, 1] = 1 / (1 + m2)
    delta[0, 0] = -c1 / (1 + m1)
    delta[0, 1] = -c2 / (1 + m2)

    return A, delta


def step_bias_calib(g_true, g_obs, verbose=False):

    fit1 = np.polyfit(g_true[:, 0], g_obs[:, 0], 1)
    fit2 = np.polyfit(g_true[:, 1], g_obs[:, 1], 1)    
    c1 = fit1[1]
    m1 = fit1[0] - 1.0
    c2 = fit2[1]
    m2 = fit2[0] - 1.0
    
    calib = (c1, m1, c2, m2)
    g_corr = step_bias_correct(g_obs, calib)    
    rmse1 = np.mean((g_true[:, 0] - g_corr[:, 0])**2)**0.5
    rmse2 = np.mean((g_true[:, 1] - g_corr[:, 1])**2)**0.5
    rmse = (rmse1, rmse2)
    
    if verbose:
        print('step biases c1: %0.3f, m1: %0.3f, c2: %0.3f, m2: %0.3f' % (c1, m1, c2, m2))
        print('rmse1: %0.3f, rmse2: %0.3f, (rmse1 + rmse2)/2: %0.3f' % (rmse[0], rmse[1], (rmse[0]+rmse[1])/2))

    return g_corr, calib, rmse


def step_bias_compose(calib1, calib2):
    
    c11, m11, c12, m12 = calib1
    c21, m21, c22, m22 = calib2
    
    m31 = m11 + m21 + m11*m21
    m32 = m12 + m22 + m12*m22

    c31 = c11 + c21*(1 + m11)
    c32 = c12 + c22*(1 + m12)
    
    calib3 = (c31, m31, c32, m32)
    return calib3