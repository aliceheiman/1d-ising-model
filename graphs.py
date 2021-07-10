from modulars import *
import numpy as np
import sys
import time
import matplotlib.pyplot as plt

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
    "figure.figsize": [10, 5],
    "axes.grid": True,
}

plt.rcParams.update(params)
plt.close("all")

#######################################
# GRAPHS
#######################################


def graph1(data, graph_params, h=False, save_fig=False):

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)

    graph_text = r""
    for param in graph_params:
        graph_text += fr"${param[0]} = {param[1]}$" + "\n"

    ax1.plot(data["x_values"], data["m_avgs"], label="Monte Carlo")

    # ax1.text(12.5, 0.70, graph_text, fontsize=15)
    ax1.text(0.77, 0.388, graph_text, fontsize=15)
    # ax1.text(-0.47, 0.405, graph_text, fontsize=15)  # H graph with energy

    if h:
        ax1.set_xlabel(r"$h$")
    else:
        ax1.set_xlabel(r"$T$")

    ax1.set_ylabel(r"$\langle m \rangle$")
    ax1.axhline(y=0, color="k", linestyle="-", linewidth=1.2)

    ax2.plot(data["x_values"], data["e_avgs"], label="Monte Carlo")

    if h:
        ax2.set_xlabel(r"$h$")
        # ax2.yaxis.set_label_position("right")
        # ax2.yaxis.tick_right()
    else:
        ax2.set_xlabel(r"$T$")
    ax2.set_ylabel(r"$\langle E \rangle$")

    if not h:
        ax2.axhline(y=0, color="k", linestyle="-", linewidth=1.2)

    # plt.tight_layout()
    plt.subplots_adjust(left=0.1, bottom=0.15, right=0.9, top=0.9, wspace=0.35, hspace=0.4)

    if save_fig:
        fig.savefig(f"_figures/{FIG_NAME}")
        print(f"Figure saved to _figures/{FIG_NAME}")
    else:
        plt.show()


def graph2(data, graph_params, mean=False, exact=False, save_fig=False):

    fig, ax = plt.subplots()

    # plt.title("System Magnetization")
    ax.plot(data["x_values"], data["m_avgs"], label="Monte Carlo")

    if mean:
        ax.plot(data["x_a_values"], data["m_calc"], label="Mean-Field")

    if exact:
        ax.plot(data["x_a_values"], data["m_exact"], label="Exact")

    ax.set_xlabel(r"$T$")
    ax.set_ylabel(r"$\langle m \rangle$")
    ax.axhline(y=0, color="k", linestyle="-", linewidth=1.2)

    if mean or exact:
        ax.legend()

    graph_text = r""
    for param in graph_params:
        graph_text += fr"${param[0]} = {param[1]}$" + "\n"

    # ax.text(12.6, 0.45, graph_text, fontsize=15)
    # ax.text(1.9, 0.44, graph_text, fontsize=15)
    ax.text(0.41, 0.22, graph_text, fontsize=15)

    # plt.tight_layout()

    if save_fig:
        fig.savefig(f"_figures/{FIG_NAME}")
        print(f"Figure saved to _figures/{FIG_NAME}")
    else:
        plt.show()


def graph3(data, graph_params, mean=False, exact=False, save_fig=False):

    fig, ax = plt.subplots()

    # plt.title("System Magnetization")
    ax.plot(data["x_values"], data["m_avgs"], label="Monte Carlo")

    if mean:
        ax.plot(data["x_a_values"], data["m_calc"], label="Mean-Field")

    if exact:
        ax.plot(data["x_a_values"], data["m_exact"], label="Exact")

    ax.set_xlabel(r"$h$")
    ax.set_ylabel(r"$\langle m \rangle$")
    ax.axhline(y=0, color="k", linestyle="-", linewidth=1.2)

    if mean or exact:
        ax.legend()

    graph_text = r""
    for param in graph_params:
        graph_text += fr"${param[0]} = {param[1]}$" + "\n"

    # ax.text(12.6, 0.45, graph_text, fontsize=15)
    # ax.text(1.9, 0.44, graph_text, fontsize=15)
    ax.text(0.41, 0.22, graph_text, fontsize=15)

    # plt.tight_layout()

    if save_fig:
        fig.savefig(f"_figures/{FIG_NAME}")
        print(f"Figure saved to _figures/{FIG_NAME}")
    else:
        plt.show()


# Load data

e_num = 20
for letter in list("a"):

    EXPERIMENT = f"experiment{e_num}{letter}"
    DATA_FILENAME = f"data/{EXPERIMENT}.npy"
    data = np.load(DATA_FILENAME, allow_pickle=True)
    data = data.item()

    # graph_params = [
    #     ("N", data["params"]["N"]),
    #     ("J", data["params"]["J"]),
    #     ("h", data["params"]["h"]),
    #     ("\lambda", data["params"]["pbc"]),
    # ]

    graph_params = [
        ("N", data["params"]["N"]),
        ("J", data["params"]["J"]),
        ("T", data["params"]["T"]),
        ("\lambda", data["params"]["pbc"]),
    ]

    FIG_NAME = f"graph{e_num}{letter}a.pdf"

    # graph3(data, graph_params, True, True, False)
    graph1(data, graph_params, False, False)
