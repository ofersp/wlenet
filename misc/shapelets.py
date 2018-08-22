import math as ma
import numpy as np
from numpy.polynomial.hermite import hermval


def basis(order, stamp_sz, beta, origin_shift = (0.5, 0.5)):

    stamp_origin = np.array(stamp_sz) / 2.0 + np.array(origin_shift)
    x, y = np.meshgrid(range(stamp_sz[0]), range(stamp_sz[1]))
    x = (x - stamp_origin[0]) / beta
    y = (y - stamp_origin[1]) / beta
    basis_ = np.zeros(np.hstack((order, stamp_sz)))
    W = np.exp(-(x**2 + y**2) / 2.0)
    for i in range(order[0]):
        for j in range(order[1]):
            coeff_x = [0]*(i+1)
            coeff_y = [0]*(j+1)
            coeff_x[i] = 1.0
            coeff_y[j] = 1.0
            Hx = hermval(x, coeff_x)
            Hy = hermval(y, coeff_y)
            norm = (2**(i+j)*ma.factorial(i)*ma.factorial(j)*ma.pi)**(-0.5)
            basis_[i, j, :, :] = Hx*Hy*W*norm/beta
    return basis_


def stamps_to_coeffs(stamps, basis_):

    stamp_sz = basis_.shape[2:]
    order = basis_.shape[:2]
    coeffs = np.zeros((stamps.shape[0], order[0], order[1]))
    for i in range(order[0]):
        for j in range(order[1]):
            coeffs[:, i, j] = np.sum(stamps*basis_[i, j, :, :].reshape(
                (1, 1, stamp_sz[0], stamp_sz[1])), axis=(2, 3))
    return coeffs


def coeffs_to_stamps(coeffs, basis_):

    stamp_sz = basis_.shape[2:]
    order = basis_.shape[:2]
    stamps = np.zeros((coeffs.shape[0], stamp_sz[0], stamp_sz[1]))

    stamps = coeffs.reshape((-1, np.prod(order))).dot(basis_.reshape((np.prod(order), np.prod(stamp_sz))))
    return stamps.reshape((coeffs.shape[0], stamp_sz[0], -1))


def operators(basis_):

    order = basis_.shape[:2]
    
    a_x = np.zeros((np.prod(order), np.prod(order)))
    a_y = np.zeros((np.prod(order), np.prod(order)))
    for i in range(1, order[0]):
        for j in range(1, order[1]):
            a_x[(i-1)*order[1] + j, i*order[1] + j] = i**0.5
            a_y[i*order[1] + (j-1), i*order[1] + j] = j**0.5

    a_x_dag = np.zeros((np.prod(order), np.prod(order)))
    a_y_dag = np.zeros((np.prod(order), np.prod(order)))           
    for i in range(0, order[0]-1):
        for j in range(0, order[1]-1):
            a_x_dag[(i+1)*order[1] + j, i*order[1] + j] = (i+1)**0.5
            a_y_dag[i*order[1] + (j+1), i*order[1] + j] = (j+1)**0.5

    S_1 = (a_x_dag.dot(a_x_dag) - a_y_dag.dot(a_y_dag) - a_x.dot(a_x) + a_y.dot(a_y)) / 2.0
    S_2 = a_x_dag.dot(a_y_dag) - a_x.dot(a_y)
    return S_1, S_2


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    order = (3, 3)
    stamp_sz = (32, 32)
    beta = 10.0
    basis_ = basis(order, stamp_sz, beta, origin_shift = (-0.5, -0.5))

    for i in range(np.prod(order)):
        plt.subplot(order[0], order[1], i+1)
        plt.imshow(basis_[int(i/3), i%3, :, :])
        plt.axis('scaled')    
    print("sum(basis_12 * basis_12) = %f" % np.sum(basis_[1, 2, :, :] * basis_[1, 2, :, :]))
    print("sum(basis_20 * basis_20) = %f" % np.sum(basis_[2, 0, :, :] * basis_[2, 0, :, :]))
    print("sum(basis_20 * basis_12) = %f" % np.sum(basis_[2, 0, :, :] * basis_[1, 2, :, :]))
    plt.show()
