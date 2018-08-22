import numpy as np
from wlenet.misc.cosmology import AngularDiameterDistance


def lensing_efficiency(cluster_redshift, background_redshifts):

    add = AngularDiameterDistance()
    zl = cluster_redshift
    zs = np.array(background_redshifts)
    dls = add.evaluate(zl, zs)
    ds = add.evaluate(0.0, zs)
    return dls/ds