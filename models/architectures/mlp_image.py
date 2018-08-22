import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Activation, Conv2D, Dropout
from keras.initializers import glorot_normal

def get_model(use_dropout=False):

    inp_shape = (32, 32, 1)
    ki = glorot_normal() # he_normal()

    model = Sequential()
    model.add(Flatten(input_shape=inp_shape))
    if use_dropout:
        model.add(Dropout(0.03))
    model.add(Dense(10, activation='relu', kernel_initializer=ki))
    if use_dropout:
        model.add(Dropout(0.05))
    model.add(Dense(50, activation='relu', kernel_initializer=ki))
    if use_dropout:
        model.add(Dropout(0.1))
    model.add(Dense(20, activation='relu', kernel_initializer=ki))
    if use_dropout:
        model.add(Dropout(0.2))
    model.add(Dense(6, name='label_pred'))

    return model