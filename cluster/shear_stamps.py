from wlenet.models.utils import load_model
from wlenet.models.predict import predict, predict_test_time_aug
from wlenet.reduction.stamps import half_light_radii, rrg_shapes
from wlenet.dataset.normalization import norm_mean_std
from wlenet.models.calibrate import step_bias_correct


class ShearEstimatorCnn(object):

    def __init__(self, model_spec, norm_func=norm_mean_std, test_time_aug=True):

        self.model_spec = model_spec
        self.norm_func = norm_func
        self.test_time_aug = test_time_aug

    def estimate(self, stamps):

        model = load_model(self.model_spec)
        x_test = self.norm_func(stamps.reshape((-1, 32, 32, 1)).copy())
        if self.test_time_aug:
            shears_cnn = predict_test_time_aug(x_test, model)
        else:
            shears_cnn = predict(x_test, model)
        shears_cnn = step_bias_correct(shears_cnn, self.model_spec['calib'])
        return shears_cnn


class ShearEstimatorRrg(object):

    def __init__(self, model_spec):

        self.model_spec = model_spec

    def estimate(self, stamps):

        psf_radius = (0.12 / 0.065) / 2.355 # TODO: get this from somewhere else
        radii = half_light_radii(stamps)
        shears_rrg = rrg_shapes(stamps, radii, psf_radius)
        shears_rrg = step_bias_correct(shears_rrg, self.model_spec['calib'])
        return shears_rrg