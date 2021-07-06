import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import random

plt.style.use("seaborn")

cmaps = ["PiYG", "PRGn", "BrBG", "PuOr", "RdGy", "RdBu", "RdYlBu", "RdYlGn", "Spectral", "coolwarm", "bwr", "seismic"]


def free_energy(j, t, n):
    return 2 * j - t * np.log(n - 1)


cmaps = ["RdYlBu", "coolwarm"]

ffunc = np.vectorize(free_energy)

cmap = "coolwarm"


T = np.linspace(0.1, 5, 100)
N = np.arange(5, 1000, 10)
Tv, Nv = np.meshgrid(T, N)


plt.suptitle("Free Energy for Temperature and System Size", fontsize=14)


for J in range(1, 7):
    deltaF = ffunc(J, Tv, Nv)

    plt.subplot(2, 3, J)

    norm = colors.TwoSlopeNorm(vmin=deltaF.min(), vcenter=0, vmax=deltaF.max())
    pc = plt.pcolormesh(Tv, Nv, deltaF, cmap=cmap, norm=norm, shading="auto")

    cbar = plt.colorbar(pc)
    cbar.ax.get_yaxis().labelpad = 15
    cbar.ax.set_ylabel("Free Energy", rotation=270)

    plt.title(f"J = {J}")
    plt.xlabel("Temperature")

    if J == 1 or J == 4:
        plt.ylabel("System Size")

plt.show()
