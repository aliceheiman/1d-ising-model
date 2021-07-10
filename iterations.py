from modulars import *
import numpy as np
import sys
import time
import matplotlib.pyplot as plt
import matplotlib as mpl

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
    # "figure.figsize": [14, 6],
    "figure.figsize": [6, 6],
    "axes.grid": True,
}

plt.rcParams.update(params)
plt.close("all")

experiment = "experiment9f"
systems = np.load(f"data/systems/{experiment}.npy", allow_pickle=True)
fig_name = f"_figures/iterations/{experiment}2.pdf"  # false
# fig_name = False

data = np.load(f"data/{experiment}.npy", allow_pickle=True)
data = data.item()

graph_params = [
    ("N", data["params"]["N"]),
    ("J", data["params"]["J"]),
    ("T", data["params"]["T"]),
    ("h", data["params"]["h"]),
    ("\lambda", data["params"]["pbc"]),
]

fig, ax = plt.subplots()

iterations, N = systems.shape

m_avgs = []

for system in systems:
    m_avgs.append(get_magnetization_density(system, N))

print(f"Avg: {np.mean(np.array(m_avgs))}")

# plt.title("System Magnetization")
ax.plot(np.arange(1, iterations + 1), m_avgs, label="Monte Carlo")

ax.set_xlabel(r"$iterations$")
ax.set_ylabel(r"$\langle m \rangle$")
ax.axhline(y=0, color="k", linestyle="-", linewidth=1.2)

graph_text = r""
for param in graph_params:
    graph_text += fr"${param[0]} = {param[1]}$" + "\n"

# ax.text(12.6, 0.45, graph_text, fontsize=15)
# ax.text(1.9, 0.44, graph_text, fontsize=15)
# ax.text(800, -0.25, graph_text, fontsize=15)
ax.text(-3300, 0.52, graph_text, fontsize=15)

fig.tight_layout()

if fig_name:
    fig.savefig(fig_name)
    print(f"Figure saved to _figures/{fig_name}")
else:
    plt.show()
