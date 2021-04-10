import argparse

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from polyagamma import random_polyagamma, polyagamma_cdf, polyagamma_pdf
from pypolyagamma import PyPolyaGamma

sns.set_style("darkgrid")

rng = np.random.default_rng(0)
pg = PyPolyaGamma(0)

data = {
    "devroye": None,
    "alternate": None,
    "gamma": None,
    "saddle": None,
    "$pypolyagamma$": None
}

def plot_kde_ecdf(h=1, z=0, size=1000, plot_cdf=False):
    for method in data:
        if method == "$pypolyagamma$":
            data[method] = [pg.pgdraw(h, z) for _ in range(size)]
        else:
            data[method] = random_polyagamma(
                h=h, z=z, method=method, size=size, random_state=rng
            )

    if plot_cdf:
        sns.ecdfplot(data=data)
    else:
        sns.kdeplot(data=data)


def plot_pdf(h, z=0, plot_cdf=False):
    xlim = 12.5
    x = np.linspace(0.01, xlim, 1000)
    f = polyagamma_cdf if plot_cdf else polyagamma_pdf

    for i in h:
        a = f(x, h=i, z=z)
        plt.plot(x, a, label=f'PG({i}, {z})')
    plt.xlabel('x')
    plt.ylabel('F(x)' if plot_cdf else 'f(x)')
    if plot_cdf:
        plt.ylim(top=1.01)
    plt.ylim(bottom=0)
    plt.xlim(right=xlim)
    plt.legend()
    plt.title(
        f'{"Distribution function" if plot_cdf else "Density function"} plot '
        f'of PG(h, {z}) for h $\in$ {1,4,7,10,15,25}.',
        fontdict = {'fontsize': 9},
    )
    plt.savefig("./cdf.svg" if plot_cdf else "./pdf.svg")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--z', default=0, type=int)
    parser.add_argument('--size', default=20000, type=int)
    parser.add_argument('--cdf', action="store_true", default=False)
    args = parser.parse_args()

    h_vals = [1, 4, 7, 10, 15, 25]
    for i in h_vals:
        plot_kde_ecdf(h=i, z=args.z, size=args.size, plot_cdf=args.cdf)

    plt.title(
        f'{"Empirical CDF" if args.cdf else "KDE"} plot of '
        f'{args.size} PG(h, {args.z}) samples using each method for '
        'h $\in$ {1,4,7,10,15,25}. \nA plot from the '
        '$pypolyagamma$ package is used as a reference.',
        fontdict = {'fontsize': 9},
    )
    plt.savefig("./ecdf.svg" if args.cdf else "./kde.svg")
    plt.figure().clear()
    plot_pdf(h_vals, z=args.z, plot_cdf=args.cdf)
