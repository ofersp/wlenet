import numpy as np

from scipy.signal import fftconvolve
from wlenet.reduction.stamps import stamp_centered_grid


class ShearFieldEstimator(object):

    def __init__(self, sse_names=['cat'], r_smoothing=30.0, scale_factor=0.1, 
                 bootstrap_iters=100, hold_out_ratio=0, 
                 seed=None, fields_cutoff=True):

        self.sse_names = sse_names
        self.r_smoothing = r_smoothing
        self.scale_factor = scale_factor
        self.bootstrap_iters = bootstrap_iters
        self.hold_out_ratio = hold_out_ratio
        self.seed = seed
        self.fields_cutoff = fields_cutoff

    def estimate(self, cluster):

        if self.seed is not None:
           np.random.seed(self.seed)

        num_sse = len(self.sse_names)

        n = cluster.cut['xy'].shape[0]
        held_out_mask = np.zeros(n).astype('bool') if self.hold_out_ratio == 0 else (np.random.rand(n) < self.hold_out_ratio)
        held_out_inds = np.where(held_out_mask)[0]
        held_in_mask = ~held_out_mask
        held_in_inds = np.where(held_in_mask)[0]

        fields = {}
        for sse_name in self.sse_names:
            fields[sse_name] = self.estimate_sse(cluster, sse_name, random_subsample=False, held_in_inds=held_in_inds, cutoff=self.fields_cutoff)
            fields_shape = fields[sse_name]['mean'].shape

        fields_cov = np.zeros((*fields_shape, num_sse, num_sse))
        for k in range(self.bootstrap_iters):
            for i in range(num_sse):
                for j in range(i, num_sse):
                    delta_i = self.estimate_sse(cluster, self.sse_names[i], random_subsample=True, 
                                                cutoff=False, held_in_inds=held_in_inds)['mean'] - fields[self.sse_names[i]]['mean']
                    delta_j = self.estimate_sse(cluster, self.sse_names[j], random_subsample=True, 
                                                cutoff=False, held_in_inds=held_in_inds)['mean'] - fields[self.sse_names[j]]['mean'] \
                                                if not i==j else delta_i
                                                
                    fields_cov[:, :, :, i, j] += delta_i * delta_j

        fields['cov'] = fields_cov / self.bootstrap_iters
        fields['held_out_inds'] = held_out_inds
        return fields

    def estimate_sse(self, cluster, sse_name, random_subsample=False, cutoff=True, min_count=5e-4, held_in_inds=None):

        kernel_sz = [151, 151]
        field_im_sz = (np.array(cluster.image_shape) * self.scale_factor).astype('int32')
        held_in_inds = held_in_inds if held_in_inds is not None else np.range(cluster.cut['xy'].shape[0])
        subsample_inds = held_in_inds if not random_subsample else \
            np.random.choice(held_in_inds, size=len(held_in_inds), replace=True)

        xx, yy = stamp_centered_grid(kernel_sz)
        kernel = np.exp(-0.5*((xx**2 + yy**2) / self.r_smoothing**2))
        kernel /= np.sum(kernel)

        stamp_x = (cluster.cut['xy'][subsample_inds, 0] * self.scale_factor + 0.5).astype('int32')
        stamp_y = (cluster.cut['xy'][subsample_inds, 1] * self.scale_factor + 0.5).astype('int32')
        stamp_shears = cluster.cut['shears_' + sse_name][subsample_inds]

        count = np.zeros(field_im_sz)
        g1_mean = np.zeros(field_im_sz)
        g2_mean = np.zeros(field_im_sz)        
        im_inds = np.ravel_multi_index((stamp_y, stamp_x), field_im_sz)

        count.ravel()[im_inds] += 1.0
        g1_mean.ravel()[im_inds] += stamp_shears[:, 0]
        g2_mean.ravel()[im_inds] += stamp_shears[:, 1]

        count = fftconvolve(count, kernel, 'same')
        g1_mean = fftconvolve(g1_mean, kernel, 'same') / count
        g2_mean = fftconvolve(g2_mean, kernel, 'same') / count

        if cutoff:
            g1_mean[count < min_count] = float('nan')
            g2_mean[count < min_count] = float('nan')

        mean  = np.append(g1_mean.reshape((1, *g1_mean.shape)), 
                          g2_mean.reshape((1, *g2_mean.shape)), axis=0)

        return {'mean': mean, 'count': count}