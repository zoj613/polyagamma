"""
This script generates optimal values for the center point ``xc`` for a given
value of the tilting parameter z. Values of ``xl`` and ``xr`` are also
calculated. These are used in the Saddle point approximation method to form
a lookup table.

For a given value z:
1) We calculate xl using the mode m = tanh(z) / z
2) Calculate xr using xr = 2 * xl.
3) Calculate xc by finding the value of x where the tangent lines L_r(x|z) and
   L_l(x|z) are equal. We find the value by finding the roots of
   f(x) = L_l(x) - L_r(x).
"""
from datetime import datetime
from math import log, sqrt, tanh, tan, cos, cosh
import numpy as np

from scipy.optimize import root_scalar

from generate_alternate_truncation_pts import pi2_8, formatted


def cgf(u, z):
    out = 0 if z == 0 else log(cosh(z))
    if u >= 0:
        s = 2 * u
        s = sqrt(s)
        out -= log(cos(s))
    else:
        s = -2 * u
        s = sqrt(s)
        out -= log(cosh(s))
    return out


def cgf_prime(u):
    if u == 0 or abs(u) < 5e-03:
        return 1
    if u > 0:
        s = sqrt(2 * u)
        return tan(s) / s
    else:
        s =sqrt(-2 * u)
        return tanh(s) / s


def ff(u, x=1):
    return cgf_prime(u) - x


if __name__ == '__main__':
    n = 100
    z_arr = np.zeros(n)
    xl_arr = np.zeros(n)
    xc_arr = np.zeros(n)
    xr_arr = np.zeros(n)
    count = 0

    for i in np.linspace(0, 8, n):
        z = 0.5 * i if i != 0 else 0
        xl = 1 if z == 0 else tanh(z) / z
        xl2 = xl * xl
        xr = 2 * xl
        half_z2 = 0 if z == 0 else 0.5 * z * z
        ul = 0 if z == 0 else -half_z2
        ur = root_scalar(ff, args=(xr,), method='ridder', bracket=[-1000, pi2_8]).root
        tr = ur + half_z2
        Lprime_l = -0.5 / xl2
        Lprime_r = -tr - 1 / xr
        Ll = lambda x: Lprime_l * x + cgf(ul, z) - 0.5 * (1/x - 1/xl) - Lprime_l * xl
        Lr = lambda x: Lprime_r * x + cgf(ur, z) - tr * xr - log(xr) + log(x) - Lprime_r * xr
        f = lambda x: Ll(x) - Lr(x)
        xc = root_scalar(f, method='ridder', bracket=[xl, 1e10]).root
        if xc > xr:
            n -= 1
            continue
        z_arr[count] = z
        xl_arr[count] = xl
        xc_arr[count] = xc
        xr_arr[count] = xr
        count += 1

    z_arr = z_arr[:n]
    xl_arr = xl_arr[:n]
    xc_arr = xc_arr[:n]
    xr_arr = xr_arr[:n]

    np.set_printoptions(formatter={'float': lambda x: "{0:0.9f}".format(x)})
    with open('./src/pgm_saddle_tangent_points.h', 'w') as f:
        f.write("/* This file is auto-generated. Do not edit by hand.\n\n")
        f.write(f"Last generated: {datetime.now()} */\n")
        f.write("\n")
        f.write(f"static const size_t pgm_saddle_tabsize = {n};")
        f.write("\n")
        f.write(f"static const double pgm_saddle_max_z = {z_arr.max()};")
        f.write("\n\n")
        f.write(formatted(z_arr, 'pgm_z'))
        f.write('\n\n')
        f.write(formatted(xl_arr, 'pgm_xl'))
        f.write('\n\n')
        f.write(formatted(xc_arr, 'pgm_xc'))
        f.write('\n\n')
        f.write(formatted(xr_arr, 'pgm_xr'))
        f.write('\n')
