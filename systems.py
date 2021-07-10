from modulars import *
import numpy as np
import sys
import time
import matplotlib.pyplot as plt
import matplotlib as mpl

plt.style.use("seaborn")
sys.setrecursionlimit(100000)

experiment = "N50system7pbc0"
systems = np.load(f"data/systems/{experiment}.npy", allow_pickle=True)

size = 50
start = 100

fig_name = f"figures/systems/{experiment}x{size}.pdf"  # false
# fig_name = False


data = np.load(f"data2/{experiment}.npy", allow_pickle=True)
data = data.item()

graph_params1 = [
    ("N", data["params"]["N"]),
    ("\lambda", data["params"]["pbc"]),
]

graph_params2 = [
    ("T", data["params"]["T"]),
    ("J", data["params"]["J"]),
    ("h", data["params"]["h"]),
]

# graph_params = []

N = data["params"]["N"]

if N == 10:
    fig_size = [3, 6]
if N == 20:
    fig_size = [3.5, 6]
if N == 50:
    if size == 50:
        fig_size = [5, 4]
    if size == 200:
        fig_size = [3.5, 7]
    if size == 300:
        fig_size = [2.8, 7]
if N == 100:
    fig_size = [4, 4]

params = {
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
    "axes.labelsize": 25,
    "legend.fontsize": 20,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "text.usetex": False,
    # "figure.figsize": [14, 6],
    "figure.figsize": fig_size,
    "axes.grid": False,
}

plt.rcParams.update(params)
plt.close("all")

# System history graph
def system_history(systems, size, graph_params1, graph_params2, start, save=False):

    graph_text1 = r""
    for param in graph_params1:
        graph_text1 += fr"${param[0]} = {param[1]}$" + "\n"

    graph_text2 = r""
    for param in graph_params2:
        graph_text2 += fr"${param[0]} = {param[1]}$" + "\n"

    color_map = {1: np.array([25, 130, 196]), -1: np.array([255, 89, 94])}  # spin up = blue  # spin down = red

    systems = systems[start : start + size]

    fig, ax = plt.subplots()

    iterations, length = systems.shape

    x = np.arange(1, length + 1)
    y = np.arange(1, iterations + 1)

    N = graph_params1[0][1]
    systems = systems.reshape(size, N)

    # make a 3d numpy array that has a color channel dimension
    data_3d = np.ndarray(shape=(systems.shape[0], systems.shape[1], 3), dtype=int)
    for i in range(0, systems.shape[0]):
        for j in range(0, systems.shape[1]):
            data_3d[i][j] = color_map[systems[i][j]]

    ax.imshow(data_3d, interpolation="nearest")
    ax.set_xlabel(r"$spin$")
    ax.set_ylabel(r"$iteration$")

    # ax.set_yticks(np.arange(size, start - 1, -1))

    c = np.array([-1, 1])
    # cmap = plt.get_cmap("jet", len(c))
    # norm = mpl.colors.BoundaryNorm([-1, 1], len(c))
    cmap = mpl.colors.ListedColormap(["#ff595e", "#1982c4"])
    norm = mpl.colors.BoundaryNorm([-1, 0, 1], cmap.N, clip=True)

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])  # this line may be ommitted for matplotlib >= 3.1

    if N == 10:
        fig.colorbar(sm, ticks=c, shrink=0.5)
        ax.text(11, 10, graph_text1, fontsize=15)
        ax.text(11, 15, graph_text2, fontsize=15)

    if N == 20:
        fig.colorbar(sm, ticks=c, shrink=0.3)
        ax.text(20, 14, graph_text1, fontsize=15)

    if N == 50:
        fig.colorbar(sm, ticks=c, shrink=0.3)

        if size == 50:
            ax.text(51, 10, graph_text1, fontsize=15)
            ax.text(51, 53, graph_text2, fontsize=15)
        if size == 200:
            ax.text(51, 20, graph_text1, fontsize=15)
            ax.text(51, 42, graph_text2, fontsize=15)
        if size == 300:
            ax.text(51, 30, graph_text1, fontsize=15)
            ax.text(51, 63, graph_text2, fontsize=15)

    if N == 100:
        fig.colorbar(sm, ticks=c, shrink=0.3)
        ax.text(20, 14, graph_text1, fontsize=15)

    plt.tight_layout()

    if save:
        print(f"Figure saved to {save}.")
        fig.savefig(save)
    else:
        plt.show()


# System mean graph
def system_mean(systems):
    pass


# Gif
def system_gif(systems):
    pass


system_history(systems, size, graph_params1, graph_params2, start, fig_name)
