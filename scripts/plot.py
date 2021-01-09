import matplotlib.pyplot as plt
import seaborn as sns

from polyagamma import default_rng

sns.set_style("darkgrid")
rng = default_rng(1)

data = {"devroye": None, "alternate": None, "gamma": None}

def plot_densities(h=1, z=0, size=1000):
    for method in data:
        data[method] = rng.polyagamma(h=h, z=z, method=method, size=size)
    sns.kdeplot(data=data)

if __name__ == "__main__":
    for i in [1, 4, 7, 10]:
        plot_densities(h=i)

    plt.title('Density plots of PG(h, 0) using each method for h $\in$ {1,4,7,10}.')
    plt.savefig("./densities.svg")
