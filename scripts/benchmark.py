import argparse

import matplotlib.pyplot as plt
import numpy as np
import perfplot
from pypolyagamma import PyPolyaGamma

from polyagamma import polyagamma


save_kwargs = {
    'transparent': True,
    'bbox_inches': 'tight',
    'logy': False,
    'logx': False,
    'time_unit': 'ms'
}


def set_val(arr, val):
    arr[:] = val
    return arr


def benchmark_methods(seed, out, z_val):
    size = out.shape[0]
    arr = np.empty(size)
    rng = np.random.default_rng(seed)
    kwargs = {'disable_checks': True, 'random_state': rng, 'out': out}

    res = perfplot.bench(
        setup=lambda n: set_val(arr, n),
        kernels=[
            lambda h: polyagamma(h, z_val, method='devroye', **kwargs),
            lambda h: polyagamma(h, z_val, method='alternate', **kwargs),
            lambda h: polyagamma(h, z_val, method='saddle', **kwargs),
            lambda h: polyagamma(h, z_val, method='gamma', **kwargs),
        ],
        labels=["devroye", "alternate", "saddle", "gamma"],
        n_range=[i for i in range(1, 60)],
        xlabel="value of $h$",
        equality_check=None,
    )
    plt.figure().suptitle(
        f"Runtime comparison of methods, generating {size} PG(h, {z_val}) "
        f"samples per value of $h$.", fontsize=9,
    )
    res.save(f"./scripts/perf_methods_{z_val}.svg", **save_kwargs)


def benchmark_samplers(seed, out, z_val=0):
    size = out.shape[0]
    arr = np.empty(size)
    z = np.empty(size)
    z[:] = z_val
    rng = np.random.default_rng(seed)
    pg = PyPolyaGamma(seed)

    kwargs = {'disable_checks': True, 'random_state': rng, 'out': out}
    res = perfplot.bench(
        setup=lambda n: set_val(arr, n),
        kernels=[
            lambda h: polyagamma(h, z_val, **kwargs),
            lambda h: pg.pgdrawv(h, z, out),
        ],
        labels=["polyagamma", "pypolyagamma"],
        n_range=[i for i in np.arange(0.1, 60, 0.4)],
        xlabel="value of $h$",
        equality_check=None,
    )
    plt.figure().suptitle(
        f"Comparison of runtime with $pypolyagamma$, generating {size} "
        f"PG(h, {z_val}) samples per value of $h$.", fontsize=11,
    )
    res.save(f"./scripts/perf_samplers_{z_val}.svg", **save_kwargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=12345, type=int)
    parser.add_argument('--size', default=20000, type=int)
    parser.add_argument('--z', default=0, type=float)
    args = parser.parse_args()

    out = np.empty(args.size)

    benchmark_methods(args.seed, out, args.z)
    benchmark_samplers(args.seed, out, args.z)
