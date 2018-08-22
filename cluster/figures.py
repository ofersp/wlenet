import numpy as np
import matplotlib.pyplot as plt

from matplotlib_scalebar.scalebar import ScaleBar
from matplotlib_scalebar.dimension import _Dimension
from wlenet.cluster.cluster_dataset_clash import ClusterDatasetClash

def show_shear_field(field, color, alpha=1.0, mini_scale_factor=None, show_scalebar=False, show_scalebar_shear=False):

    g1_im_ = field['mean'][0, :, :]
    g2_im_ = field['mean'][1, :, :]
    
    # Translate shear components into sines and cosines
    data = np.zeros([2, *(g1_im_[0::20, 0::20].shape)])
    data[0, :, :] = g1_im_[0::20, 0::20]
    data[1, :, :] = g2_im_[0::20, 0::20]

    g_mag = np.sqrt(data[0]**2 + data[1]**2)
    cos_2_phi = data[0] / g_mag
    sin_2_phi = data[1] / g_mag

    # Compute stick directions
    cos_phi = np.sqrt(0.5*(1.0 + cos_2_phi)) * np.sign(sin_2_phi)
    sin_phi = np.sqrt(0.5*(1.0 - cos_2_phi))

    # Fix ambiguity when sin_2_phi = 0
    cos_phi[sin_2_phi==0] = np.sqrt(0.5*(1.0 + cos_2_phi[sin_2_phi==0]))
    
    plt.quiver(g_mag*cos_phi/2, -g_mag*sin_phi/2, headlength=0, headwidth=0, headaxislength=0, 
               scale_units='xy', scale=0.075, color=color, width=0.006, alpha=alpha)
    plt.quiver(-g_mag*cos_phi/2, g_mag*sin_phi/2, headlength=0, headwidth=0, headaxislength=0, 
               scale_units='xy', scale=0.075, color=color, width=0.006, alpha=alpha)

    if show_scalebar:

        clash = ClusterDatasetClash()
        mini_pixel_scale_arcmin = 20 * clash.get_pixel_scale() * mini_scale_factor**-1 * 60**-1
        scalebar = ScaleBar(dx=mini_pixel_scale_arcmin, dimension=_Dimension('m', latexrepr="'"))
        plt.gca().add_artist(scalebar)

    if show_scalebar_shear:

        shear_scale = 100*0.075
        scalebar = ScaleBar(dx=shear_scale, dimension=_Dimension('m', latexrepr="%"))
        plt.gca().add_artist(scalebar)        


def show_survey_shear_fields(survey, sse_name, sfe_name='r_smoothing_30'):
    
    survey_mini_scale_factor = survey.cluster_template.mini_scale_factor

    fig = plt.figure(figsize=(12,10.25))
    #fig.tight_layout()

    for i, cluster in enumerate(survey.clusters):
        plt.subplot(4, 5, i+1)
        plt.gca().invert_yaxis()
        show_shear_field(cluster.shear_fields[sfe_name][sse_name], color='b', alpha=0.75, 
        	mini_scale_factor=survey_mini_scale_factor,
        	show_scalebar=(i==0), show_scalebar_shear=(i==1))

        plt.title(cluster.full_name)
        plt.xticks([], [])
        plt.yticks([], [])

    plt.tight_layout()