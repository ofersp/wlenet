import math as ma
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


def abline(a, b, style_str='--', alpha=1.0):
    
    """Plot a line from slope and intercept"""
    axes = plt.gca()
    xlim_kept = axes.get_xlim()
    ylim_kept = axes.get_ylim()
    x_vals = np.array(axes.get_xlim())
    y_vals = b + a * x_vals
    plt.plot(x_vals, y_vals, style_str, alpha=alpha)
    axes.set_xlim(xlim_kept)
    axes.set_ylim(ylim_kept)


def plot_ellipses(x, y, sigma_xx, sigma_yy, sigma_xy, factor=2, ax=None, alpha=1.0):

    if ax is None:
        ax = plt.gca()

    # compute the ellipse principal axes and rotation from covariance
    theta = 0.5 * np.arctan2(2 * sigma_xy, (sigma_xx - sigma_yy))
    tmp1 = 0.5 * (sigma_xx + sigma_yy)
    tmp2 = np.sqrt(0.25 * (sigma_xx - sigma_yy) ** 2 + sigma_xy ** 2)
    sigma1, sigma2 = np.sqrt(tmp1 + tmp2), np.sqrt(tmp1 - tmp2)

    for i in range(len(x)):
        ax.add_patch(Ellipse((x[i], y[i]), 
                             factor * sigma1[i], factor * sigma2[i],
                             theta[i] * 180.0 / np.pi, fc='none', ec='k', alpha=alpha))