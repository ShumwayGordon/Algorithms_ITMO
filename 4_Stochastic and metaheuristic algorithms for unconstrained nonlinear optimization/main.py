import math
import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.optimize import dual_annealing
from scipy.optimize import differential_evolution
from scipy.optimize import minimize
from anneal import SimAnneal


def f(x):
    return 1 / (x ** 2 - 3 * x + 2)


def generate_data(N):
    k = range(0, N+1)
    x = [0] * (N+1)
    y = [0] * (N+1)
    sigma = np.random.standard_normal(N+1)
    for i in range(N+1):
        x[i] = (3 * k[i]) / 1000
        fx = f(x[i])
        if fx < -100:
            y[i] = -100 + sigma[i]
        elif fx >= -100 and fx <= 100:
            y[i] = fx + sigma[i]
        elif fx > 100:
            y[i] = 100 + sigma[i]

    return x, y


def approx_ratio(x, a, b, c, d):
    return (a * x + b) / (x ** 2 + c * x + d)


def num_minim(x, y, a, b, c, d):
    D = 0
    for k in range(len(y)):
        D += (approx_ratio(x[k], a, b, c, d) - y[k]) ** 2
    return D


def num_minimize_vec(point, x, y):
    a, b, c, d = point
    D = 0
    for k in range(len(y)):
        D += (approx_ratio(x[k], a, b, c, d) - y[k]) ** 2
    return D


def minim(x0, x, y):
    a, b, c, d = x0
    D = 0
    res = [(a * xx + b) / (xx ** 2 + c * xx + d) - yy for xx, yy in zip(x, y)]
    return res


def nelder_mead(x, y, prec=1e-3):
    x0 = np.array([-1.0, 1.0, -2.0, 1.0])

    res = minimize(num_minimize_vec, x0, args=(x, y), tol=prec, method='Nelder-Mead')

    return res.x[0], res.x[1], res.x[2], res.x[3], res.nfev, res.nit, num_minim(x, y, res.x[0], res.x[1],
                                                                                      res.x[2], res.x[3])


def leven_marq(x, y, prec=1e-3):
    x0 = np.array([-1.0, 1.0, -2.0, 1.0])

    res = least_squares(minim, x0, args=(x, y), ftol=prec, xtol = prec, method='lm')

    return res.x[0], res.x[1], res.x[2], res.x[3], res.nfev, res.nfev // 2, num_minim(x, y, res.x[0], res.x[1],
                                                                                      res.x[2], res.x[3])


def annealing(x, y, prec=1e-3):
    x0 = [-1.0, 1.0, -2.0, 1.0]
    lw = [-10] * 4
    up = [10] * 4

    res = dual_annealing(num_minimize_vec, args=(x, y), x0 = x0, restart_temp_ratio=prec, bounds=list(zip(lw, up)))

    return res.x[0], res.x[1], res.x[2], res.x[3], res.nfev, res.nfev // 2, num_minim(x, y, res.x[0], res.x[1],
                                                                                      res.x[2], res.x[3])


def diff_evolution(x, y, prec=1e-3):
    x0 = [-1.0, 1.0, -2.0, 1.0]
    lw = [-10] * 4
    up = [10] * 4

    res = differential_evolution(num_minimize_vec, args=(x, y), x0 = x0, bounds=list(zip(lw, up)))

    return res.x[0], res.x[1], res.x[2], res.x[3], res.nfev, res.nfev // 2, num_minim(x, y, res.x[0], res.x[1],
                                                                                      res.x[2], res.x[3])


def read_coords(path):
    coords = []
    with open(path, "r") as f:
        for line in f.readlines():
            line = [float(x.replace("\n", "")) for x in line.split(" ")]
            #print(line)
            coords.append(line)
    return coords


def main():
    """
    precision = 0.001
    N = 1000
    x, y, = generate_data(N)

    a_1, b_1, c_1, d_1, it_1, fc_1, dd_1 = nelder_mead(x, y, precision)
    a_2, b_2, c_2, d_2, it_2, fc_2, dd_2 = leven_marq(x, y, precision)
    a_3, b_3, c_3, d_3, it_3, fc_3, dd_3 = annealing(x, y, precision)
    a_4, b_4, c_4, d_4, it_4, fc_4, dd_4 = diff_evolution(x, y, precision)

    print(dd_1, a_1, b_1, c_1, d_1, it_1, fc_1)
    print(dd_2, a_2, b_2, c_2, d_2, it_2, fc_2)
    print(dd_3, a_3, b_3, c_3, d_3, it_3, fc_3)
    print(dd_4, a_4, b_4, c_4, d_4, it_4, fc_4)

    y_aprox_1 = [(a_1 * xx + b_1) / (xx ** 2 + c_1 * xx + d_1) for xx in x]
    y_aprox_2 = [(a_2 * xx + b_2) / (xx ** 2 + c_2 * xx + d_2) for xx in x]
    y_aprox_3 = [(a_3 * xx + b_3) / (xx ** 2 + c_3 * xx + d_3) for xx in x]
    y_aprox_4 = [(a_4 * xx + b_4) / (xx ** 2 + c_4 * xx + d_4) for xx in x]

    plt.plot(x, y, 'k-', x, y_aprox_1, 'r--', x, y_aprox_2, 'g--', x, y_aprox_3, 'b--', x, y_aprox_4, 'y--')
    plt.title('Approximation of generated data by rational function')
    plt.legend(['data', 'Nelder-Mead', 'Leven-Marq', 'Annealing', 'Evolution'], loc='best')
    plt.grid(True)
    plt.show()

    """

    ################################## task 2 ##################################
    coords = read_coords("sgb128_coords.txt")  # generate_random_coords(100)
    sa = SimAnneal(coords, stopping_iter=10000)
    sa.anneal()
    sa.visualize_routes()
    sa.visualize_first()
    sa.plot_learning()


if __name__ == "__main__":
    main()