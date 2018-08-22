import numpy as np
import importlib
import keras

from os.path import expanduser, isfile
from wlenet import config
from pprint import PrettyPrinter


def get_output_dim(model):

    return model.get_layer('label_pred').get_weights()[0].shape[-1]


def load_model(model_spec, load_weights=True, show_summary=False):
    
    model_weights_path = expanduser(config['trained_models_path'] + '/weights/' + model_spec['trained_name'] + '.hdf5')
    model_arch = importlib.import_module('wlenet.models.architectures.' + model_spec['arch_name'])
    model = model_arch.get_model(**model_spec['kwargs_arch'])
    if load_weights:
        assert(isfile(model_weights_path))
        model.load_weights(model_weights_path)
    if show_summary:
        model.summary()
    return model


def save_model(model_spec, model):
    
    model_weights_path = expanduser(config['trained_models_path'] + '/weights/' + model_spec['trained_name'] + '.hdf5')
    model.save_weights(model_weights_path)


def save_spec(model_spec):

    trained_name = model_spec['trained_name']
    model_spec_path = expanduser(config['trained_models_path'] + '/specs/' + trained_name + '.npy')
    np.save(model_spec_path, model_spec)


def load_spec(trained_name):
    
    model_spec_path = expanduser(config['trained_models_path'] + '/specs/' + trained_name + '.npy')
    model_spec = np.load(model_spec_path).item()
    return model_spec


def print_spec(*args, **kwargs):
    
    pp = PrettyPrinter(width=200)
    pp.pprint(*args, **kwargs)


def freeze_upper_layers(model):

    for l in model.layers:
        if isinstance(l, keras.layers.convolutional.Conv2D):
            l.trainable = False
        if isinstance(l, keras.layers.core.Dense):
            l.trainable = False
        if isinstance(l, keras.layers.core.Dropout):
            l.trainable = False
            l.rate = 0.0
            
    return model