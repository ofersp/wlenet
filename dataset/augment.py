import numpy as np


def transpose_stamps(stamps):
    return stamps.transpose((0, 2, 1, 3))


def horz_flip_stamps(stamps):
    return np.flip(stamps, 1)


def vert_flip_stamps(stamps):
    return np.flip(stamps, 2)


def stamp_shear_symm_aug(stamps):

    stamps_aug = np.zeros((8, *stamps.shape))
    shear_signs = np.zeros((8, 2))

    shear_signs[0, :] = [+1, +1]; stamps_aug[0, ...] = stamps
    shear_signs[1, :] = [-1, +1]; stamps_aug[1, ...] = transpose_stamps(stamps_aug[0, ...])
    shear_signs[2, :] = [-1, -1]; stamps_aug[2, ...] = horz_flip_stamps(stamps_aug[1, ...])
    shear_signs[3, :] = [+1, -1]; stamps_aug[3, ...] = transpose_stamps(stamps_aug[2, ...])
    shear_signs[4, :] = [+1, +1]; stamps_aug[4, ...] = vert_flip_stamps(stamps_aug[3, ...])
    shear_signs[5, :] = [-1, +1]; stamps_aug[5, ...] = transpose_stamps(stamps_aug[4, ...])
    shear_signs[6, :] = [-1, -1]; stamps_aug[6, ...] = horz_flip_stamps(stamps_aug[5, ...])
    shear_signs[7, :] = [+1, -1]; stamps_aug[7, ...] = transpose_stamps(stamps_aug[6, ...])
    
    return stamps_aug, shear_signs