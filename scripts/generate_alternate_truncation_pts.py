"""
This script generates optimal truncation points for each value of `h` parameter
in the interval [1, 4]. The values are written into a C header file called
src/pgm_alternate_trunc_points.h. These are intended to be used when sampling
from a Polya-gamma distribution using the Alternate method. Values of h in this
interval not present in the table are calculated using a linear interpolation.

We find the roots the function f(x) =  l(x|h) - r(x|h) for a given value of h.
"""
from datetime import datetime
from math import log, sqrt, pi, exp, cosh
import numpy as np

from scipy.optimize import root_scalar
from scipy.special import gamma, loggamma

log2 = log(2)
twopi = 2 * pi
logpi_2 = log(0.5 * pi)
pi2_8 = 0.125 * (pi ** 2)


def l(x, h):
    return exp(h * log2 + log(h) - 0.5 * log(twopi * x * x * x) - 0.5 * (h ** 2) / x)


def l_prime(x, h):
    h2 = h * h
    return (2 ** h) * h / sqrt(twopi) * exp(-0.5 * h2 / x) * (-1.5 * (x ** (-2.5))
            + 0.5 * h2 * (x ** (-3.5)))


def r(x, h):
    return exp(0.5 * h * logpi_2 + (h - 1) * log(x) - pi2_8 * x - loggamma(h))


def r_prime(x, h):
    return exp(h * logpi_2 - loggamma(h) - pi2_8 * x) * ((h - 1) * (x ** (h -
        2)) - pi2_8 * (x ** (h - 1)))


def f(x, h=1):
    return l(x, h) - r(x, h)


def f_prime(x, h):
    return l_prime(x, h) - r_prime(x, h)


def chunk(arr, size=3):
    """return an iterable of arr values in chunks of `size`"""
    return (arr[i:i + size] for i in range(0, len(arr), size))


def stringify_chunk(chunk):
    out = str(chunk)
    out = out.replace("[", '')
    out = out.replace("]", '')
    out = out.replace(" ", ', ')
    return out


def formatted(arr, name):
    nl = '\n'
    out = f'static const double {name}[] = {{'
    line_fmt = '    {},'.format
    for i in chunk(arr, 5):
        string_i = stringify_chunk(i)
        out = nl.join([out, line_fmt(string_i)])
    # trim the trailing comma
    out = out[:-1]
    out = nl.join([out, '};\n'])
    return out


if __name__ == "__main__":
    n = 50
    h_val = np.zeros(n)
    f_val = np.zeros(n)
    count = 0
    # find roots of f given each value of h in [1, 4]
    for i in np.linspace(1, 4, n):
        sol = root_scalar(f, args=(i,), bracket=[0.001, 20]).root
        h_val[count] = i
        f_val[count] = sol
        count += 1

    h_range = h_val.max() - h_val.min()

    np.set_printoptions(formatter={'float': lambda x: "{0:0.9f}".format(x)})
    with open('./src/pgm_alternate_trunc_points.h', 'w') as f:
        f.write("/* This file is auto-generated. Do not edit by hand.\n\n")
        f.write(f"Last generated: {datetime.now()} */\n")
        f.write("\n")
        f.write(f"static const size_t pgm_table_size = {n};\n")
        f.write(f"static const double pgm_h_range = {h_range};")
        f.write("\n\n")
        f.write(formatted(h_val, 'pgm_h'))
        f.write('\n\n')
        f.write(formatted(f_val, 'pgm_f'))
        f.write('\n')
