"""
This script generates optimal truncation points for each value of `h` parameter
in the interval [1, 4]. The values are written into a C header file called
src/pgm_alternate_trunc_points.h. These are intended to be used when sampling
from a Polya-gamma distribution using the Alternate method. Values of h in this
interval not present in the table are calculated using a linear interpolation.

We minimize the function f(x) =  p(x|h) + q(x|h) for a given value of h.
"""
import argparse
from datetime import datetime
from math import log, pi, exp
import numpy as np

from scipy.optimize import minimize_scalar
from scipy.special import gammaincc

log2 = log(2)
log4_pi = log(4 / pi)


def f(x, h=1):
    return (exp(h * log2) * gammaincc(0.5, 0.5 * h * h / x) +
            exp(h * log4_pi) * gammaincc(h, x * pi * pi / 8))


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
    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose', action='store_true', default=False)
    parser.add_argument('--size', type=int, default=25)
    parser.add_argument('--maxh', type=int, default=4)
    parser.add_argument('--scale', type=float, default=2)
    args = parser.parse_args()
    n = args.size
    h_val = np.zeros(n)
    f_val = np.zeros(n)
    count = 0
    for i in np.linspace(1, args.maxh, n):
        sol = minimize_scalar(f, method='bounded', bounds=[1e-200, 10000], args=(i,)).x
        if args.verbose:
            print(f"h={i} | root={sol} | f(root)={f(sol)} | f(scaled_root)={f(args.scale * sol)}")
        h_val[count] = i
        f_val[count] = args.scale * sol
        count += 1

    np.set_printoptions(formatter={'float': lambda x: "{0:0.9f}".format(x)})
    with open('./src/pgm_alternate_trunc_points.h', 'w') as f:
        f.write("/* This file is auto-generated. Do not edit by hand.\n\n")
        f.write(f"Last generated: {datetime.now()} */\n")
        f.write("\n")
        f.write(f"static const size_t pgm_table_size = {n};\n")
        f.write(f"static const size_t pgm_maxh = {args.maxh};")
        f.write("\n\n")
        f.write(formatted(h_val, 'pgm_h'))
        f.write('\n\n')
        f.write(formatted(f_val, 'pgm_f'))
        f.write('\n')
