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
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "text.usetex": True,
    "figure.figsize": [3.44, 3.44],
    "axes.grid": False,
}

plt.rcParams.update(params)
plt.close("all")

#######################################
# SETUP
#######################################

iteration = 1000
thermalization = 0.3
epsilon = 1.0e-6
mean_field = True
energy_graph = False
exact = True

offset_x = 0  # -0.021
offset_y = 0  # 0.040
bottom_right = [
    (0.81 + offset_x, 0.24 + offset_y),
    (0.81 + offset_x, 0.21 + offset_y),
    (0.81 + offset_x, 0.18 + offset_y),
]
top_right = [(0.81, 0.84), (0.81, 0.80), (0.81, 0.76)]
bottom_left = [(0.15, 0.22), (0.15, 0.18), (0.15, 0.14)]
top_left = [(0.15, 0.84), (0.15, 0.80), (0.15, 0.76)]
# top_left = [(0.13, 0.84), (0.13, 0.82), (0.13, 0.80)]

graph = 1
positioning = top_right

ITERATIONS, T_ITERATIONS = get_iterations(iteration, thermalization)
x_label, x_values = get_xlabel("parameters.txt")

# Save data
m_avgs = []
m_stds = []
e_avgs = []
e_stds = []

m_guess = np.random.rand()
m_calc = []
m_exact = []

#######################################
# MAIN
#######################################

for N, J, T, h, has_jt, pbc in get_parameters("parameters.txt"):

    start = time.time()

    if has_jt:
        J_t = has_jt
    else:
        J_t = J / T

    # Save values
    m_data = []
    e_data = []

    # Create a system
    system, even_vec, odd_vec = initialize_system(N)

    for i in range(ITERATIONS):

        # Perform flips
        dE = 2 * J_t * even_vec * (odd_vec + np.hstack((odd_vec[1:], pbc * odd_vec[0]))) + 2 * h * even_vec
        dE = dE.astype(np.float64)
        accepted_flips = get_accepted_flips(dE, N, T, has_jt)
        even_vec[accepted_flips] *= -1

        dE = 2 * J_t * odd_vec * (even_vec + np.hstack((pbc * even_vec[-1], even_vec[:-1]))) + 2 * h * odd_vec
        dE = dE.astype(np.float64)
        accepted_flips = get_accepted_flips(dE, N, T, has_jt)
        odd_vec[accepted_flips] *= -1

        # Do not save data for thermalization
        if i < T_ITERATIONS:
            continue

        # compute magnetization and energy
        m_data.append(get_magnetization_density(system, N))
        e_data.append(get_energy_density(system, pbc, J_t, h, N))

    # Perform bootstrapping
    m_avg, m_std = bootstrap(N, m_data)
    e_avg, e_std = bootstrap(N, e_data)

    # Save the data
    m_avgs.append(m_avg)
    m_stds.append(m_std)
    e_avgs.append(e_avg)
    e_stds.append(e_std)

    # Get analytical values and save them
    m_mean_field = mean_field_approximation(m_guess, J, T, h, epsilon)
    m_calc.append(m_mean_field)

    # Get exact values
    m_exact.append(get_exact_solution(h, J, T))

    end = time.time()

    # Output information
    print(f"Done in {end - start}s")
    print(f"\tFree Energy: {get_free_energy(J, T, N)}")
    print(f"\tCritical Temperature: {get_critical_temperature(J, T, N)}")
    print(f"\tJ_t: {J_t}, h: {h}")
    print(f"\tm_avg: {m_avg}")
    print(f"\te_avg: {e_avg}")

#######################################
# GRAPHS
#######################################

# Convert into numpy arrays
m_avgs = np.array(m_avgs)
m_stds = np.array(m_stds)
e_avgs = np.array(e_avgs)
e_stds = np.array(e_stds)

m_calc = np.array(m_calc)

# MAGNETIZATION

if energy_graph:
    plt.subplot(1, 2, 1)

plt.plot(x_values, m_avgs, label="Monte Carlo")

if mean_field:
    plt.plot(x_values, m_calc, label="Mean-Field Approximation")

if exact:
    plt.plot(x_values, m_exact, label="Exact Solution")

plt.title("System Magnetization")

if graph == 1 or graph == 2:
    params = [("N", N), ("J", J), ("h", h)]
if graph == 3:
    params = [("N", N), ("J", J), ("T", T)]

for param, position in zip(params, positioning):
    plt.figtext(position[0], position[1], f"{param[0]} = {param[1]}")

plt.xlabel(x_label)
plt.ylabel("<m>")

plt.axhline(y=0, color="k", linestyle="-", linewidth=1.2)

if mean_field or exact:
    plt.legend()

lower_bound = m_avgs - m_stds
upper_bound = m_avgs + m_stds
plt.fill_between(x_values, lower_bound, upper_bound, alpha=0.3)

if energy_graph:
    # ENERGY

    plt.subplot(1, 2, 2)
    plt.plot(x_values, e_avgs, label="Monte Carlo")
    plt.title("System Energy")
    plt.xlabel(x_label)
    plt.ylabel("<E>")

    lower_bound = e_avgs - e_stds
    upper_bound = e_avgs + e_stds
    plt.fill_between(x_values, lower_bound, upper_bound, alpha=0.3)

# Display graph!
plt.show()

# fig.savefig("B_complete.pdf",bbox_inches='tight')
