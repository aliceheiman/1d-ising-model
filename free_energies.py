import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import random
import sys

plt.style.use("seaborn")
sys.setrecursionlimit(100000)

params = {
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
    "axes.labelsize": 25,
    "legend.fontsize": 20,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "text.usetex": False,
    "figure.figsize": [10, 6.5],
    # "figure.figsize": [10, 5],
    "axes.grid": True,
}

plt.rcParams.update(params)
plt.close("all")

cmaps = ["PiYG", "PRGn", "BrBG", "PuOr", "RdGy", "RdBu", "RdYlBu", "RdYlGn", "Spectral", "coolwarm", "bwr", "seismic"]


def free_energy(j, t, n):
    return 2 * j - t * np.log(n - 1)


cmaps = ["RdYlBu", "coolwarm"]

ffunc = np.vectorize(free_energy)

cmap = "coolwarm"


T = np.linspace(0.1, 5, 100)
N = np.arange(5, 1000, 10)
Tv, Nv = np.meshgrid(T, N)

fig, axes = plt.subplots(nrows=2, ncols=3)

J = 1
for ax in axes.flat:
    deltaF = ffunc(J, Tv, Nv)
    norm = colors.TwoSlopeNorm(vmin=deltaF.min(), vcenter=0, vmax=deltaF.max())
    pc = ax.pcolormesh(Tv, Nv, deltaF, cmap=cmap, norm=norm, shading="auto")

    ax.set_title(fr"$J={J}$", fontsize=22)

    # Hide y-axis
    if J in [2, 3, 5, 6]:
        ax.get_yaxis().set_visible(False)
    else:
        ax.set_ylabel(r"$N$")

    # Hide x-axis
    if J in [1, 2, 3]:
        ax.get_xaxis().set_visible(False)
    else:
        ax.set_xlabel(r"$T$")

    J += 1

# plt.tight_layout()
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
fig.colorbar(pc, cax=cbar_ax)

# plt.show()
plt.savefig("_figures/free_energies_b.pdf")
print("Figure saved.")

sys.exit()

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
    plt.xlabel("T")

    if J == 1 or J == 4:
        plt.ylabel("N")

plt.show()
