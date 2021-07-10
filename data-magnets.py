from modulars import *
import numpy as np
import sys
import time
import matplotlib.pyplot as plt

plt.style.use("seaborn")
sys.setrecursionlimit(100000)

#######################################
# SETUP
#######################################

iteration = 1000
thermalization = 0.3
epsilon = 1.0e-6
mean_field = True
energy_graph = False
exact = True
save_system = False
save_experiment = True
absolute = False
analytical = True
quadratic = True

EXPERIMENT = "graph13"
DATA_FILENAME = f"data2/{EXPERIMENT}.npy"


ITERATIONS, T_ITERATIONS = get_iterations(iteration, thermalization)
x_label, x_values, x_a_values = get_xlabel("parameters.txt")

# Save data
m_avgs = []
m_stds = []
e_avgs = []
e_stds = []

m_calc = []
m_exact = []

specific_heats = []
susceptibilities = []

systems = []

#######################################
# MAIN
#######################################

print("Starting Monte Carlo Simulation...")

p_params, a_params = get_parameters("parameters.txt")

current_iteration = 1
total_iterations = len(p_params)

# COMPUTE MONTE CARLO
for N, J, T, h, has_jt, pbc in p_params:

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

    if save_system:
        systems.append(system.copy())
        print(system)

    for i in range(ITERATIONS):

        # Perform flips
        dE = 2 * J_t * even_vec * (odd_vec + np.hstack((odd_vec[1:], pbc * odd_vec[0]))) + 2 * h * even_vec
        dE = dE.astype(np.float64)
        accepted_flips = get_accepted_flips(dE, N, T, has_jt)
        even_vec[accepted_flips] *= -1

        # Save a snapshot of system
        if save_system:
            systems.append(system.copy())

        dE = 2 * J_t * odd_vec * (even_vec + np.hstack((pbc * even_vec[-1], even_vec[:-1]))) + 2 * h * odd_vec
        dE = dE.astype(np.float64)
        accepted_flips = get_accepted_flips(dE, N, T, has_jt)
        odd_vec[accepted_flips] *= -1

        # Do not save data for thermalization
        if i < T_ITERATIONS:
            continue

        # compute magnetization and energy
        m_data.append(get_magnetization_density(system, N))
        e_data.append(get_energy_density(system, pbc, J, h, N))
        # e_data.append(get_energy_v2(T, J, N, h))

    # Perform bootstrapping
    m_avg, m_std = bootstrap(N, m_data)
    e_avg, e_std = bootstrap(N, e_data)

    # if absolute:
    #     m_avg = np.mean(np.absolute(np.array(m_data)))
    # else:
    #     m_avg = np.mean(np.array(m_data))

    if quadratic:
        m_avg = np.mean(np.array(m_data) ** 2)
    else:
        m_avg = np.mean(np.array(m_data))

    e_avg = np.mean(e_data)

    # Save the data
    m_avgs.append(m_avg)
    m_stds.append(m_std)
    e_avgs.append(e_avg)
    e_stds.append(e_std)

    specific_heats.append(specific_heat(m_data, T))
    susceptibilities.append(susceptibility(m_data, T))

    end = time.time()

    # Output information
    print(f"Iteration {current_iteration}/{total_iterations} in {end - start}s")
    print(f"\tFree Energy: {get_free_energy(J, T, N)}")
    print(f"\tCritical Temperature: {get_critical_temperature(J, T, N)}")
    print(f"\tJ_t: {J_t}, h: {h}")
    print(f"\tm_avg: {m_avg}")
    print(f"\te_avg: {e_avg}")

    current_iteration += 1

# ANALYTICAL APPROACH

if analytical:
    print("Starting analytical computations...")

    current_iteration = 1
    total_iterations = len(a_params)

    start = time.time()

    for N, J, T, h, has_jt, pbc in a_params:

        m_guess = get_exact_solution(h, J, T)
        # m_guess = np.tanh(h / T)

        start = time.time()

        if has_jt:
            J_t = has_jt
        else:
            J_t = J / T

        # Get analytical values and save them
        m_calc.append(mean_field_approximation(m_guess, J, T, h, epsilon))

        # Get exact values
        m_exact.append(get_exact_solution(h, J, T))

    end = time.time()

    print(f"Analytical computations done in {end - start}s...")

#######################################
# SAVE DATA
#######################################

print("Saving data...")

if save_system:
    systems = np.array(systems)
    np.save(f"data/systems/{EXPERIMENT}.npy", systems)

    print(f"Saved system snapshots to data/systems/{EXPERIMENT}.npy")

if save_experiment:

    # Convert into numpy arrays
    m_avgs = np.array(m_avgs)
    m_stds = np.array(m_stds)
    e_avgs = np.array(e_avgs)
    e_stds = np.array(e_stds)

    m_calc = np.array(m_calc)
    m_exact = np.array(m_exact)

    # Create big data structure
    data = {
        "x_values": x_values,
        "x_label": x_label,
        "x_a_values": x_a_values,
        "m_avgs": m_avgs,
        "m_stds": m_stds,
        "e_avgs": e_avgs,
        "e_stds": e_stds,
        "m_calc": m_calc,
        "m_exact": m_exact,
        "heats": specific_heats,
        "suscept": susceptibilities,
        "params": {"N": N, "J": J, "T": T, "h": h, "pbc": pbc, "epsilon": epsilon, "iteration": iteration},
    }

    # Save dictionary
    np.save(DATA_FILENAME, data)

    print(f"Saved experiment data to {DATA_FILENAME}.")
