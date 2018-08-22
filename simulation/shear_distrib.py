import numpy as np


def uniform(g_min, g_max):

    g1 = np.random.rand(1)[0] * (g_max - g_min) + g_min
    g2 = np.random.rand(1)[0] * (g_max - g_min) + g_min
    return g1, g2


def truncated_normal(g_std, g_norm_max):

    done = False
    while not done:
        g1, g2 = np.random.randn(2) * g_std
        done = (g1**2 + g2**2) <= g_norm_max**2        
    return g1, g2