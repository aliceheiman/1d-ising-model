from modulars import *
import numpy as np
import sys

#######################################
# PARAMETERS
#######################################
thermalization = 0.3
epsilon = 1.0e-6

ITERATIONS, T_ITERATIONS = get_iterations(1000, thermalization)
x_label, x_values = get_xlabel("parameters.txt")

# Save data
m_avgs = []
m_stds = []
e_avgs = []
e_stds = []

for N, J, T, h, beta, pbc in get_parameters("parameters.txt"):
    if beta:
        J_t = beta
    else:
        J_t = J / T

    # Save values
    m_data = []
    e_data = []

    # Create a system
    system, even_vec, odd_vec = initialize_system(N)

    for i in range(ITERATIONS):

        # Perform flips
        n_flip(even_vec, odd_vec, J_t, pbc, h, N, T, beta)

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

    print(f"\tJ_t: {J_t}, h: {h}")
    print(f"\tm_avg: {m_avg}")
    print(f"\te_avg: {e_avg}")

# Create graph
