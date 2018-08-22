# Adapted from: https://github.com/michetonu/gradient_reversal_keras_tf

import math as ma
import tensorflow as tf
from keras.engine import Layer
import keras.backend as K


def reverse_gradient(X, hp_lambda):
    '''Flips the sign of the incoming gradient during training.'''
    try:
        reverse_gradient.num_calls += 1
    except AttributeError:
        reverse_gradient.num_calls = 1

    grad_name = "GradientReversal%d" % reverse_gradient.num_calls

    @tf.RegisterGradient(grad_name)
    def _flip_gradients(op, grad):
        return [tf.negative(grad) * hp_lambda]

    g = K.get_session().graph
    with g.gradient_override_map({'Identity': grad_name}):
        y = tf.identity(X)

    return y


class GradientReversal(Layer):

    default_name = "gradient_reversal"

    '''Flip the sign of gradient during training.'''
    def __init__(self, lambda_init=0.0, gamma=10.0, **kwargs):
        super(GradientReversal, self).__init__(**kwargs)
        self.supports_masking = False
        self.hp_lambda = K.variable(value=lambda_init, dtype='float32', name='gradient_reversal_hp_lambda')
        self.gamma = gamma

    def build(self, input_shape):
        self.trainable_weights = []

    def call(self, x, mask=None):
        return reverse_gradient(x, self.hp_lambda)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {'hp_lambda': self.get_lambda(), 'gamma': self.gamma}
        base_config = super(GradientReversal, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def get_lambda(self):
        return K.get_value(self.hp_lambda)

    def set_lambda(self, hp_lambda):
        return K.set_value(self.hp_lambda, hp_lambda)

    def update_lambda(self, epoch, total_epochs, verbose=True):        
        
        p = float(epoch) / float(total_epochs)
        hp_lambda_new = 2.0 / (1.0 + ma.exp(-self.gamma*p)) - 1.0    
        self.set_lambda(hp_lambda_new)
        if verbose:
            print('GradientReversal layer lambda updated to: ' + str(self.get_lambda()))