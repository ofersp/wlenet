import numpy as np
import matplotlib.pyplot as plt


def show_first_conv_kernels(model):

    l = model.get_layer(index=1)
    w = l.get_weights()
    plt.rcParams["figure.figsize"] = [6, 6]
    for i in range(64):
        plt.subplot(8, 8, i+1)
        plt.imshow(w[0][:,:,0,i])
        plt.axis('off')
    plt.show()


def show_first_dot_products(model):

    l = model.get_layer(index=3)
    w = l.get_weights()
    for i in range(w[0].shape[1]):
        plt.subplot(2, 5, i+1)
        plt.imshow(w[0][:, i].reshape(32, 32))
        plt.axis('off')
    plt.show()


def show_scatter_label_pred(y, y_pred, name='', name_pred='',
                            min_g=-0.2, max_g=0.2, min_g_true=-0.2, max_g_true=0.2, 
                            alpha=0.15, figsize=[12, 6]):

    fig = plt.figure(figsize=figsize)

    plt.subplot(1, 2, 1)
    plt.plot(y[:, 0], y_pred[:, 0], '.', alpha=alpha)
    plt.plot([min_g, max_g], [min_g, max_g], 'k', alpha=0.5)
    plt.axis([min_g_true, max_g_true, min_g, max_g])
    plt.xlabel(name + ' $g_1$')
    plt.ylabel(name_pred + ' $g_1$')
    plt.grid('on')

    plt.subplot(1, 2, 2)
    plt.plot(y[:, 1], y_pred[:, 1], '.', alpha=alpha)
    plt.plot([min_g, max_g], [min_g, max_g], 'k', alpha=0.5)
    plt.axis([min_g_true, max_g_true, min_g, max_g])
    plt.xlabel(name + ' $g_2$')
    plt.ylabel(name_pred + ' $g_2$')
    plt.grid('on')