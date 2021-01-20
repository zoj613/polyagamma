import argparse

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from polyagamma import polyagamma

sns.set_style("darkgrid")
rng = np.random.default_rng(1)

data = {"devroye": None, "alternate": None, "gamma": None, "saddle": None}

def plot_densities(h=1, z=0, size=1000):
    for method in data:
        data[method] = polyagamma(h=h, z=z, method=method, size=size, random_state=rng)
    sns.kdeplot(data=data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--z', default=0, type=float)
    args = parser.parse_args()

    for i in [1, 4, 7, 10, 15]:
        plot_densities(h=i, z=args.z)

    plt.title(
        f'Density plots of PG(h, {args.z})' +
        ' using each method for h $\in$ {1,4,7,10,15}.'
    )
    plt.savefig("./densities.svg")
