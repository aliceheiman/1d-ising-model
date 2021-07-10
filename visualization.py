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

ITERATIONS, T_ITERATIONS = get_iterations(iteration, thermalization)
x_label, x_values = get_xlabel("parameters.txt")

N = 1000
J = 1
T = 10
h = 0.5
pbc = 0
has_jt = False


systems = np.random.choice([1, -1], (10, 5))
print(systems)

iterations, spins = systems.shape

# make color chart
x = np.arange(1, spins + 1)
y = np.arange(1, iterations + 1)

plt.pcolormesh(x, y, systems, shading="auto", cmap="turbo")
plt.colorbar()
plt.show()

sys.exit()


#######################################
# MAIN
#######################################

if has_jt:
    J_t = has_jt
else:
    J_t = J / T

# Save values
m_data = []
e_data = []
systems = []

# Create a system
system, even_vec, odd_vec = initialize_system(N)
systems.append(system.copy())

for i in range(ITERATIONS):

    # Perform flips
    dE = 2 * J_t * even_vec * (odd_vec + np.hstack((odd_vec[1:], pbc * odd_vec[0]))) + 2 * h * even_vec
    dE = dE.astype(np.float64)
    accepted_flips = get_accepted_flips(dE, N, T, has_jt)
    even_vec[accepted_flips] *= -1

    systems.append(system.copy())

    dE = 2 * J_t * odd_vec * (even_vec + np.hstack((pbc * even_vec[-1], even_vec[:-1]))) + 2 * h * odd_vec
    dE = dE.astype(np.float64)
    accepted_flips = get_accepted_flips(dE, N, T, has_jt)
    odd_vec[accepted_flips] *= -1

    systems.append(system.copy())

    # Do not save data for thermalization
    if i < T_ITERATIONS:
        continue

    # compute magnetization and energy
    m_data.append(get_magnetization_density(system, N))
    e_data.append(get_energy_density(system, pbc, J_t, h, N))

# Get visual image


# Perform bootstrapping
m_avg, m_std = bootstrap(N, m_data)
e_avg, e_std = bootstrap(N, e_data)
