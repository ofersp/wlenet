import numpy as np

from keras.utils import Sequence
from wlenet.dataset.noise import correlated_noise_stamps
from wlenet.dataset.normalization import norm_mean_std


class CorrelatedNoiseSequence(Sequence):

    def __init__(self, source, noise_kernel, target=None, 
                 batch_size=100, batch_size_source=50, 
                 epoch_factor=0.1, norm_func=norm_mean_std, 
                 aug_noise_factor=False, aug_noise_factor_min=0.1, aug_noise_factor_max=1.0):
        
        self.aug_noise_factor = aug_noise_factor
        self.aug_noise_factor_min = aug_noise_factor_min
        self.aug_noise_factor_max = aug_noise_factor_max
        self.epoch_factor = epoch_factor
        self.curr_epoch = 0        
        self.source = source
        self.target = target
        self.n_source, l, m, self.stamp_ch = source['x'].shape
        self.n_target = 0 if target is None else target['x'].shape[0]
        self.label_dim = source['y'].shape[1]
        self.batch_size = batch_size
        self.batch_size_source = batch_size if target is None else batch_size_source
        self.batch_size_target = batch_size - batch_size_source
        self.norm_func = norm_func
        self.noise_kernel = noise_kernel
        self.stamp_sz = np.array([l, m])

        assert(self.stamp_ch == 1)        
        assert(len(source['x']) == len(source['y']))
        assert(self.batch_size > 0)
        assert(self.batch_size_source >= 0)
        assert(self.batch_size_target >= 0)

        if self.target is not None:
            assert(source['x'].shape[1:] == target['x'].shape[1:])

    def getitem_at_epoch(self, batch_idx, epoch_idx):

        seed_factor_index = 100321  # sequence randomness is a function of these constants
        seed_factor_noise = 100543  # and (batch_idx, epoch_idx)

        batch_ind = epoch_idx * self.__len__() + batch_idx

        seed_index = (batch_ind * seed_factor_index) % (2**32)
        seed_noise = (batch_ind * seed_factor_noise) % (2**32)

        np.random.seed(seed_index) # TODO: make sure this seeding does not conflict other PRNGs in lensing

        inds_source = np.random.choice(self.n_source, self.batch_size_source, replace=True)  
        x_batch_source = self.source['x'][inds_source, :, :, :]
        y_batch_source = self.source['y'][inds_source, :]
        x_noise_source = correlated_noise_stamps(len(inds_source), self.stamp_sz, self.noise_kernel, seed_noise, 
                                                 self.aug_noise_factor, self.aug_noise_factor_min, self.aug_noise_factor_max)
        x_batch_source = x_batch_source + x_noise_source

        if self.target is not None:

            inds_target = np.random.choice(self.n_target, self.batch_size_target, replace=True)
            x_batch_target = self.target['x'][inds_target, :, :, :]
            y_batch_target = np.zeros((self.batch_size_target, self.label_dim)) # target labels are all zeros

            x_batch = np.concatenate([x_batch_source, x_batch_target])
            y_batch = np.concatenate([y_batch_source, y_batch_target])

            y_batch[:,[2]] = np.concatenate([np.ones ((self.batch_size_source, 1)),   # this encodes the sample weights in the loss
                                             np.zeros((self.batch_size_target, 1))])  # as the third column of y

            d_batch = np.concatenate([np.tile([1, 0], [self.batch_size_source, 1]),   # these encode the domain labels
                                      np.tile([0, 1], [self.batch_size_target, 1])])  # source first, target second

        else:
            inds_target = np.array([])
            x_batch = x_batch_source
            y_batch = y_batch_source

        x_batch = x_batch if not self.norm_func else self.norm_func(x_batch)    

        labels = [y_batch, d_batch] if self.target is not None else y_batch
        return x_batch, labels, inds_source, inds_target

    def __getitem__(self, idx):

        return self.getitem_at_epoch(idx, self.curr_epoch)[:2]

    def __len__(self):

        return int(self.epoch_factor * self.n_source / self.batch_size)

    def on_epoch_end(self):

        self.curr_epoch += 1

    def reset(self):

        self.curr_epoch = 0


def generate_batches(seq, num_batches, batch_idx_start=0, epoch_idx=0):

    x = np.zeros((num_batches, seq.batch_size, *seq.stamp_sz, seq.stamp_ch))
    y = np.zeros((num_batches, seq.batch_size, seq.label_dim))                      # the regression labels
    d = None if seq.target is None else np.zeros((num_batches, seq.batch_size, 2))  # the domain labels
    inds = np.zeros((num_batches, seq.batch_size))

    for i in range(num_batches):

        x_, y_, inds_source, inds_target = seq.getitem_at_epoch(batch_idx_start + i, epoch_idx)
        x[i, :, :, :, :] = x_
        inds[i, :] = np.concatenate([inds_source, inds_target])

        if seq.target is None:
            y[i, :, :] = y_
        else:
            y[i, :, :] = y_[0]
            d[i, :, :] = y_[1]

    x = x.reshape(-1, *seq.stamp_sz, seq.stamp_ch)
    y = y.reshape(-1, seq.label_dim)
    d = None if seq.target is None else d.reshape(-1, 2)
    inds = inds.reshape(-1, 1)
    
    return (x, y, inds) if seq.target is None else (x, y, inds, d)