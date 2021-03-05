import argparse

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from polyagamma import polyagamma
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

def plot_densities(h=1, z=0, size=1000):
    for method in data:
        if method == "$pypolyagamma$":
            data[method] = [pg.pgdraw(h, z) for _ in range(size)]
        else:
            data[method] = polyagamma(
                h=h, z=z, method=method, size=size, random_state=rng
            )
    sns.kdeplot(data=data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--z', default=0, type=int)
    parser.add_argument('--size', default=20000, type=int)
    args = parser.parse_args()

    for i in [1, 4, 7, 10, 15, 25]:
        plot_densities(h=i, z=args.z, size=args.size)

    plt.title(
        f'Density plots of {args.size} PG(h, {args.z}) samples' +
        ' using each method for h $\in$ {1,4,7,10,15,25}. \nA plot from the ' +
        '$pypolyagamma$ package is used as a reference.',
        fontdict = {'fontsize': 9},
    )
    plt.savefig("./densities.svg")
