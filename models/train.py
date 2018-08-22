import keras
import numpy as np
import keras.backend as K

from keras.utils import Sequence
from os.path import expanduser
from wlenet import config
from wlenet.models.losses import gaussian_nll, gaussian_nll_weighted
from wlenet.models.utils import load_model, save_spec, freeze_upper_layers
from wlenet.models.gradient_reversal import GradientReversal


def train_model(model_spec, set_train, set_test,
                label_loss=None, optim_func=keras.optimizers.rmsprop, 
                epochs=100, batch_size=100, lr_init=3e-4, min_lr=2e-6, reduce_factor=0.1**0.5,
                show_summary=False, generator_workers=8, patience=10, dry_run=False, fine_tune=False):

    assert(isinstance(set_train, Sequence))

    use_discriminator = model_spec['kwargs_arch']['use_discriminator']

    known_label_losses = {'gaussian_nll': gaussian_nll, 'gaussian_nll_weighted': gaussian_nll_weighted}
    label_loss = known_label_losses.get(label_loss, label_loss)

    trained_models_path = expanduser(config['trained_models_path'])
    weights_path = trained_models_path + '/weights/' + model_spec['trained_name'] + '.hdf5'
    log_path = trained_models_path + '/logs/' + model_spec['trained_name'] + '.csv'    
    
    optim = optim_func(lr=lr_init)
    model = load_model(model_spec, load_weights=fine_tune, show_summary=show_summary)
    if fine_tune:
        model = freeze_upper_layers(model)

    if use_discriminator:
        model.compile(loss={'label_pred': label_loss, 'domain_pred': 'categorical_crossentropy'},
                      metrics={'domain_pred': 'accuracy'},
                      optimizer=optim)
        monitor = 'val_label_pred_loss'
    else:
        model.compile(loss=label_loss, optimizer=optim)
        monitor = 'val_loss'    

    lr_reducer = keras.callbacks.ReduceLROnPlateau(factor=reduce_factor, cooldown=0, patience=patience, min_lr=min_lr, verbose=True)
    model_chkp = keras.callbacks.ModelCheckpoint(weights_path, save_best_only=True, monitor=monitor)
    csv_logger = keras.callbacks.CSVLogger(log_path)

    if dry_run:
        callbacks = [lr_reducer]
    else:
        callbacks = [model_chkp, lr_reducer, csv_logger]

    if use_discriminator:
        grl_update_lambda = keras.callbacks.LambdaCallback(on_epoch_end=lambda epoch, logs: 
                                                           model.get_layer(name = GradientReversal.default_name).update_lambda(epoch, epochs))
        callbacks.append(grl_update_lambda)

    if not dry_run:
        save_spec(model_spec)

    model.fit_generator(generator=set_train, validation_data=set_test,
                        epochs=epochs, verbose=2, shuffle=False, callbacks=callbacks,
                        max_queue_size=10, use_multiprocessing=True, workers=generator_workers)
    return model