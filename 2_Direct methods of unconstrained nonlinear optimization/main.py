import math
import numpy as np
import random
import matplotlib.pyplot as plt


def f1(x):
    return pow(x, 3) # [0, 1]


def f2(x):
    return abs(x - 0.2) # [0, 1]


def f3(x):
    return x * math.sin(1/x) # [0.01, 1]


def aprox_linear(x, a, b):
    return a * x + b


def aprox_ratio(x, a, b):
    if b*x == -1:
        b+=0.0001
    return a / (1 + b * x)


def num_minimize(func, x, y, a, b):
    D = 0
    for k in range(len(y)):
        D += (func(x[k], a, b) - y[k])**2
    return D


def num_minimize_vec(point, func, x, y):
    a, b = point
    D = 0
    for k in range(len(y)):
        D += (func(x[k], a, b) - y[k])**2
    return D


def min_exhaustive(func, start, end, precision):
    iterations = 0
    func_calc = 0
    min_cur = float(func(start))
    x_min = float(start)

    for x in np.arange(start, end + precision * 0.001, precision):
        iterations += 1
        temp = func(x)
        func_calc += 1
        if temp < min_cur:
            min_cur = temp
            x_min = x

    return min_cur, x_min, iterations, func_calc


def min_dichotomy(func, start, end, precision):
    iterations = 0
    func_calc = 0
    b = end
    a = start
    while (b - a) / 2 >= precision:
        iterations += 1
        m = (a + b) / 2
        c = m - (precision / 2)
        d = m + (precision / 2)

        if func(c) < func(d):
            b = d
        else:
            a = c
        func_calc += 2
    min_cur = func((b+a) / 2)
    func_calc += 1
    return min_cur, (b+a) / 2, iterations, func_calc


def min_golden(func, start, end, precision):
    iterations = 0
    func_calc = 0
    # phi = (1 + math.sqrt(5)) / 2.0
    b = end
    a = start

    c = a + ((3 - math.sqrt(5)) / 2) * (b - a)
    d = b + ((math.sqrt(5) - 3) / 2) * (b - a)

    fc = func(c)
    fd = func(d)
    func_calc += 2

    while b - a >= precision:
        iterations += 1
        if fc < fd:
            b = d
            d = c
            fd = fc
            c = a + ((3 - math.sqrt(5)) / 2 ) * (b - a)
            fc = func(c)
            func_calc += 1
        else:
            a = c
            c = d
            fc = fd
            d = b + ((math.sqrt(5) - 3) / 2) * (b - a)
            fd = func(d)
            func_calc += 1

    min_cur = func((a + b) / 2)
    func_calc += 1
    return min_cur, (a + b) / 2, iterations, func_calc


def min_golden_2(func, x, y, start, end, arg_fix, arg_ind, precision):
    # func - approximation func (linear or rational)
    # start, end - first & second element for searching best a (or b)
    # arg_fix, arg_ind - if arg_ind = 1, we calc 'a' and arg_fix is  fixed 'b' (and contrary)
    iterations = 0
    func_calc = 0
    b = end
    a = start
    c = a + ((3 - math.sqrt(5)) / 2) * (b - a)
    d = b + ((math.sqrt(5) - 3) / 2) * (b - a)

    if arg_ind:
        fc = num_minimize(func, x, y, c, arg_fix)  # if arg_ind = 1 -> we calc 'a' param
        fd = num_minimize(func, x, y, d, arg_fix)
    else:
        fc = num_minimize(func, x, y, arg_fix, c)  # if arg_ind = 1 -> we calc 'b' param
        fd = num_minimize(func, x, y, arg_fix, d)
    func_calc += 2

    while b - a >= precision:
        iterations += 1
        if fc < fd:
            b = d
            d = c
            fd = fc
            c = a + ((3 - math.sqrt(5)) / 2 ) * (b - a)
            if arg_ind:
                fc = num_minimize(func, x, y, c, arg_fix)
            else:
                fc = num_minimize(func, x, y, arg_fix, c)
            func_calc += 1
        else:
            a = c
            c = d
            fc = fd
            d = b + ((math.sqrt(5) - 3) / 2) * (b - a)
            if arg_ind:
                fd = num_minimize(func, x, y, d, arg_fix)
            else:
                fd = num_minimize(func, x, y, arg_fix, d)
            func_calc += 1

    if arg_ind:
        min_cur = num_minimize(func, x, y, (a + b) / 2.0, arg_fix)
    else:
        min_cur = num_minimize(func, x, y, arg_fix, (a + b) / 2.0)
    func_calc += 1
    # min_cur - minimum D with found a(or b) and fixed b(or a)
    # (a + b) / 2 - best found value of a(or b)
    return min_cur, (a + b) / 2, iterations, func_calc


def min_2_exhaustive(func, y, x, start, end, prec):
    D = num_minimize(func, x, y, 0, 0)
    best_a = 0
    best_b = 0
    iterations = 0
    func_calcs = 0
    # step = 0.1
    for a in np.arange(start, end, prec):
        for b in np.arange(start, end, prec):
            iterations += 1
            func_calcs += 1
            temp = num_minimize(func, x, y, a, b)
            if temp < D: ############################ need D(i+1) - D() < precision
                D = temp
                best_a = a
                best_b = b
    return D, best_a, best_b, iterations, func_calcs


def min_2_hessian(func, y, x, start, end, prec):
    d = num_minimize(func, x, y, 0, 0)
    iterations_while = 0
    iterations_golden = 0
    func_calcs = 0
    best_a = 0
    best_b = 0
    step = 0.1
    while True:
        iterations_while += 1
        d_cur, best_a, iter_cur, func_calc = min_golden_2(func, x, y, start, end, best_b, 1, step)
        func_calcs += func_calc
        iterations_golden += iter_cur
        if round(abs(d - d_cur), 3) < prec:
            d = d_cur
            break
        d = d_cur

        d_cur, best_b, iter_cur, func_calc = min_golden_2(func, x, y, start, end, best_a, 0, step)
        func_calcs += func_calc
        iterations_golden += iter_cur
        if abs(d - d_cur) < prec:
            d = d_cur
            break
        else:
            step = step / 2.0
            d = d_cur
    return d, best_a, best_b, iterations_while, iterations_golden, func_calcs


class Vector(object):
    def __init__(self, x, y):
        """ Create a vector, example: v = Vector(1,2) """
        self.x = x
        self.y = y
    def __repr__(self):
        return "({0}, {1})".format(self.x, self.y)
    def __add__(self, other):
        x = self.x + other.x
        y = self.y + other.y
        return Vector(x, y)
    def __sub__(self, other):
        x = self.x - other.x
        y = self.y - other.y
        return Vector(x, y)
    def __rmul__(self, other):
        x = self.x * other
        y = self.y * other
        return Vector(x, y)
    def __truediv__(self, other):
        x = self.x / other
        y = self.y / other
        return Vector(x, y)
    def c(self):
        return (self.x, self.y)


def min_2_nelder_mead(func, y, x, prec):
    alpha = 4
    betta = 2
    gamma = 8
    func_calcs = 0
    # initialization
    v1 = Vector(0.5, 0.5)
    v2 = Vector(0.5, 0.0)
    v3 = Vector(0.0, 0.5)
    d = num_minimize_vec(v3.c(), func, x, y)
    func_calcs += 1
    best_a, best_b = v3.c()
    iterations = 0
    old_b = v3

    while True:
        iterations += 1
        adict = {v1:num_minimize_vec(v1.c(), func, x, y), v2:num_minimize_vec(v2.c(), func, x, y), v3:num_minimize_vec(v3.c(), func, x, y)}

        func_calcs += 3
        points = sorted(adict.items(), key=lambda xx: xx[1])

        b = points[0][0]
        g = points[1][0]
        w = points[2][0]
        mid = (g + b)/2

        a_old, b_old = old_b.c()
        a_new, b_new = b.c()

        if a_new != a_old and b_new != b_old:
            print(old_b, b)
            d = num_minimize_vec(old_b.c(), func, x, y)
            new_d = num_minimize_vec(b.c(), func, x, y)
            if round(abs(d - new_d), 3) < prec:
                d = new_d
                best_a, best_b = b.c()
                break

        # reflection
        xr = mid + alpha * (mid - w)
        func_calcs += 2
        if num_minimize_vec(xr.c(), func, x, y) < num_minimize_vec(g.c(), func, x, y):
            w = xr
        else:
            func_calcs += 2
            if num_minimize_vec(xr.c(), func, x, y) < num_minimize_vec(w.c(), func, x, y):
                w = xr
            c = (w + mid)/2
            func_calcs += 2
            if num_minimize_vec(c.c(), func, x, y) < num_minimize_vec(w.c(), func, x, y):
                w = c
        func_calcs += 2
        if num_minimize_vec(xr.c(), func, x, y) < num_minimize_vec(b.c(), func, x, y):
            # expansion
            xe = mid + gamma * (xr - mid)
            func_calcs += 2
            if num_minimize_vec(xe.c(), func, x, y) < num_minimize_vec(xr.c(), func, x, y):
                w = xe
            else:
                w = xr
        func_calcs += 2
        if num_minimize_vec(xr.c(), func, x, y) > num_minimize_vec(g.c(), func, x, y):
            # contraction
            xc = mid + betta * (w - mid)
            func_calcs += 2
            if num_minimize_vec(xc.c(), func, x, y) < num_minimize_vec(w.c(), func, x, y):
                w = xc
        # update points
        v1 = w
        v2 = g
        v3 = b
        old_b = b
        func_calcs += 1
        #temp_d = num_minimize_vec(v3.c(), func, x, y)
        #print(abs(d - temp_d))
        # best_a, best_b = v3.c()
        #if abs(d - temp_d) < prec:
        #    best_a, best_b = v3.c()
        #    d = temp_d
        #    break
        #d = temp_d

    return d, best_a, best_b, iterations, func_calcs

#print(min_exhaustive(f3, 0.01, 1, 0.001))
#print(min_dichotomy(f3, 0.01, 1, 0.001))
#print(min_golden(f3, 0.01, 1, 0.001))

#print(min_exhaustive(f2, 0, 1, 0.001))
#print(min_dichotomy(f2, 0, 1, 0.001))
#print(min_golden(f2, 0, 1, 0.001))


################################ part 2 ################################

precision = 0.001

a = random.uniform(0.0, 1.0)
b = random.uniform(0.0, 1.0)

k = range(0, 101)

x = [0] * 101
y = [0] * 101
sigma = np.random.standard_normal(101)

for kk in k:
    x[kk] = kk / 100
    y[kk] = a * x[kk] + b + sigma[kk]

d_1, a_1, b_1, iterations_1, func_calcs_1 = min_2_exhaustive(aprox_linear, y, x, -0.9, 1, precision) # 0.5069999999996138 0.586999999999605 123.65071503845361

d_2, a_2, b_2, iterations_while, iterations_golden, func_calcs_2 = min_2_hessian(aprox_linear, y, x, -0.9, 1, precision)

d_3, a_3, b_3, iterations_3, func_calcs_3 = min_2_nelder_mead(aprox_linear, y, x, precision)

print(d_1, a_1, b_1, iterations_1, func_calcs_1)
print(d_2, a_2, b_2, iterations_while, iterations_golden, func_calcs_2)
print(d_3, a_3, b_3, iterations_3, func_calcs_3)

y_aprox_1 = [a_1 * xx + b_1 for xx in x]
y_aprox_2 = [a_2 * xx + b_2 for xx in x]
y_aprox_3 = [a_3 * xx + b_3 for xx in x]

plt.plot(x, y, 'k-', x, y_aprox_1, 'r--', x, y_aprox_2, 'g--', x, y_aprox_3, 'b--')
plt.title('Approximation of generated data by linear function')
plt.legend(['data', 'exhaustive', 'Gauss', 'Nelder-Mead'], loc='best')
plt.grid(True)
plt.show()


