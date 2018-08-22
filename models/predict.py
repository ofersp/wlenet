import numpy as np
from wlenet.dataset.augment import stamp_shear_symm_aug


def predict(stamps, model):

    preds = model.predict(stamps)
    shears = preds[0] if isinstance(preds, list) else preds
    return shears


def predict_test_time_aug(stamps, model):

    stamps_aug, shear_signs = stamp_shear_symm_aug(stamps)

    shears = np.zeros((8, stamps.shape[0], 2))
    for i in range(8):
        shears[i, :, :] = predict(stamps_aug[i, ...], model)[:, :2] * shear_signs[[i], :]

    shears = np.mean(shears, axis=0)    
    return shears