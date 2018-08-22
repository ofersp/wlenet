import numpy as np
import matplotlib.pyplot as plt


def display_catalog_histograms(gss, just_valid = False):

    inds = gss.valid_indices if just_valid else np.arange(gss.rgc.nobjects)

    plt.subplot(1,3,1)
    plt.hexbin(gss.fit_catalog['mag_auto'][inds], gss.fit_catalog['zphot'][inds], gridsize=25, bins='log')
    plt.title('mag_auto vs. zphot')
    plt.colorbar()
    plt.subplot(1,3,2)
    plt.hexbin(gss.fit_catalog['mag_auto'][inds], gss.fit_catalog['flux_radius'][inds], gridsize=25, bins='log')
    plt.title('mag_auto vs. flux_radius')
    plt.colorbar()
    plt.subplot(1,3,3)
    plt.hexbin(gss.fit_catalog['zphot'][inds], gss.fit_catalog['flux_radius'][inds], gridsize=25, bins='log')
    plt.title('zphot vs. flux_radius')
    plt.colorbar()
    plt.show()