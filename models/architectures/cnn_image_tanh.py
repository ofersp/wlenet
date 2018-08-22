import keras

from keras.models import Model
from keras.layers import Input, Conv2D, Flatten, Dense, Dropout
from keras.initializers import glorot_normal
from wlenet.models.gradient_reversal import GradientReversal


def get_model(inp_shape=(32, 32, 1), out_dim=6, first_conv_size=(5, 5), first_conv_strides=(3, 3), num_chan=(64, 40, 10, 100, 30),
              use_dropout_conv=False,  dropout_conv_rate=0.2, 
              use_dropout_dense=False, dropout_dense_rate=0.2,              
              use_discriminator=False, grl_lambda_init=0.0, grl_gamma=10.0, grl_input_varname=None):

    ki = glorot_normal()

    inputs = Input(shape=inp_shape, name='inputs')
    conv_1 = Conv2D(num_chan[0], first_conv_size, strides=first_conv_strides, activation='relu', kernel_initializer=ki, name='conv_1')(inputs)
    conv_1_do = Dropout(dropout_conv_rate)(conv_1) if use_dropout_conv else conv_1 
    conv_2 = Conv2D(num_chan[1], (1, 1), strides=(1, 1), activation='relu', kernel_initializer=ki, name='conv_2')(conv_1_do)
    conv_2_do = Dropout(dropout_conv_rate)(conv_2) if use_dropout_conv else conv_2 
    conv_3 = Conv2D(num_chan[2], (1, 1), strides=(1, 1), activation='relu', kernel_initializer=ki, name='conv_3')(conv_2_do)
    conv_3_do = Dropout(dropout_conv_rate)(conv_3) if use_dropout_conv else conv_3
    flatten = Flatten(name='flatten')(conv_3_do)
    dense_1 = Dense(num_chan[3], activation='relu', kernel_initializer=ki, name='dense_1')(flatten)
    dense_1_do = Dropout(dropout_dense_rate)(dense_1) if use_dropout_dense else dense_1
    dense_2 = Dense(num_chan[4], activation='tanh', kernel_initializer=ki, name='dense_2')(dense_1_do)
    dense_2_do = Dropout(dropout_dense_rate)(dense_2) if use_dropout_dense else dense_2
    label_pred = Dense(out_dim, kernel_initializer=ki, name='label_pred')(dense_2_do)

    if use_discriminator:
        assert(grl_input_varname is not None)
        domain_grl = GradientReversal(lambda_init=grl_lambda_init, gamma=grl_gamma, name=GradientReversal.default_name)(locals()[grl_input_varname])
        domain_dense_1 = Dense(100, activation='relu', name='domain_dense_1')(domain_grl)
        domain_dense_2 = Dense(50, activation='relu', name='domain_dense_2')(domain_dense_1)
        domain_pred = Dense(2, activation='softmax', name='domain_pred')(domain_dense_2)
        outputs = [label_pred, domain_pred]

    outputs = [label_pred, domain_pred] if use_discriminator else label_pred
    model = Model(inputs=inputs, outputs=outputs)
    return model