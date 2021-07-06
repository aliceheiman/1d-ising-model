import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import random

plt.style.use("seaborn")


def get_critical_temperature(j, n):
    return (2 * j) / (np.log(n - 1))


cfunc = np.vectorize(get_critical_temperature)


def approach1():
    J = np.linspace(1, 5, 100)
    N = np.arange(5, 1000, 10)
    Jv, Nv = np.meshgrid(J, N)

    critical_temps = cfunc(Jv, Nv)

    pc = plt.pcolormesh(Jv, Nv, critical_temps, cmap="coolwarm", shading="auto")

    cbar = plt.colorbar(pc)
    cbar.ax.get_yaxis().labelpad = 15
    cbar.ax.set_ylabel("Critial Temperature", rotation=270)

    # plt.title(f"J = {J}")
    plt.xlabel("Temperature")

    plt.show()


def approach2():

    N = np.arange(5, 1000)
    plt.suptitle("Critical Temperature for Different System Sizes", fontsize=14)

    for J in range(1, 7):
        critical_temps = cfunc(J, N)

        plt.subplot(2, 3, J)
        plt.plot(N, critical_temps)
        plt.ylim(0, 5)
        plt.title(f"J = {J}", fontsize=12)
        plt.xlabel("System Size")

        if J == 1 or J == 4:
            plt.ylabel("Critical Temperature")

    plt.show()


approach2()
