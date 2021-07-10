import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from numpy.random import sample

#######################################
# HELPER FUNCTIONS
#######################################
def output_image(system, filename):
    """
    Generates a pixel image from an Ising Model system,
    composed of 1's (rendered blue) and -1's (rendered red).
    """

    height, width = system.shape
    cell_size = 50
    cell_border = 2

    # Create a blank canvas
    img = Image.new("RGBA", (width * cell_size, height * cell_size), "black")
    draw = ImageDraw.Draw(img)

    for i, row in enumerate(system):
        for j, col in enumerate(row):

            # Spin up
            if col == 1:
                fill = (25, 130, 196)

            # Spin down
            elif col == -1:
                fill = (255, 89, 94)

            # Empty cell
            else:
                fill = (237, 240, 252)

            # Draw cell
            draw.rectangle(
                (
                    [
                        (j * cell_size + cell_border, i * cell_size + cell_border),
                        ((j + 1) * cell_size - cell_border, (i + 1) * cell_size - cell_border),
                    ]
                ),
                fill=fill,
            )

    img.save(filename)


def get_accepted_flips(dE, N, T, has_jt=False):
    """Calculates accepted spins based on the Boltzmann distribution."""
    if not has_jt:
        return np.random.rand(N // 2) < np.minimum(np.ones(N // 2), np.exp((-dE / T)))
    else:
        return np.random.rand(N // 2) < np.minimum(np.ones(N // 2), np.exp((-dE)))


def get_magnetization_density(system, N):
    """Computes the magnetization density of a system using the formula <M>/N."""
    return np.sum(system) / N


def get_energy_density(system, pbc, J, h, N):
    """Computes the energy density of a certain system."""
    system_shifted = np.hstack((system[1:], pbc * system[0]))

    return (-J * np.sum(system * system_shifted) - h * np.sum(system)) / N


def get_energy_v2(T, J, N, h):
    def lambda1(J, B, h):
        return np.exp(B * J) * np.cosh(B * h) + np.sqrt(np.exp(2 * B * J) * np.sinh(B * h) ** 2 + np.exp(-2 * B * J))

    def labmda_root(J, B, h):
        return np.sqrt(np.exp(2 * B * J) * np.sinh(B * h) ** 2 + np.exp(-2 * B * J))

    B = 1 / T
    x = B * h
    l1 = lambda1(J, B, h)
    a = np.sqrt(labmda_root(J, B, h))

    energy = (-T / l1) * (
        np.exp(B * J) * (J * np.cosh(x) + h * np.sinh(x))
        + (np.exp(2 * B * J) / a) * (J * np.sinh(x) ** 2 + h * np.sinh(x) * np.cosh(x) - 2 * J * np.exp(-4 * B * J))
    )

    return energy


def n_flip(even_vec, odd_vec, J_t, pbc, h, N, T, has_jt):
    """Performs an odd and an even flip on a system."""
    dE = 2 * J_t * even_vec * (odd_vec + np.hstack((odd_vec[1:], pbc * odd_vec[0]))) + 2 * h * even_vec
    accepted_flips = get_accepted_flips(dE, N, T, has_jt)
    even_vec[accepted_flips] *= -1

    dE = 2 * J_t * odd_vec * (even_vec + np.hstack((pbc * even_vec[-1], even_vec[:-1]))) + 2 * h * odd_vec
    accepted_flips = get_accepted_flips(dE, N, T, has_jt)
    odd_vec[accepted_flips] *= -1


def even_flip(even_vec, odd_vec, J_t, pbc, h, N, T):
    """Performs an even flip on a system."""
    dE = 2 * J_t * even_vec * (odd_vec + np.hstack((odd_vec[1:], pbc * odd_vec[0]))) + 2 * h * even_vec
    accepted_flips = get_accepted_flips(dE)
    even_vec[accepted_flips] *= -1


def odd_flip(even_vec, odd_vec, J_t, pbc, h, N, T):
    """Performs an odd flip on a system."""
    dE = 2 * J_t * odd_vec * (even_vec + np.hstack((pbc * even_vec[-1], even_vec[:-1]))) + 2 * h * odd_vec
    accepted_flips = get_accepted_flips(dE, N, T)
    odd_vec[accepted_flips] *= -1


def specific_heat(data, T):
    return (np.std(data) / T) ** 2


def susceptibility(data, T):
    return (np.std(data) ** 2) / T


def bootstrap(N, data):
    """Performs the bootstrapping technique on a list of data to get a standard deviation."""

    # Parameters
    sample_size = len(data)
    samples = N * 100

    # Get the average of 'samples' number of samples with 'sample_size' elements in each sample
    means = np.mean(np.random.choice(data, (samples, sample_size)), axis=1)
    # means = [np.mean(np.random.choice(data, size=len(data), replace=True)) for i in range(samples)]

    # Return the average and standard deviation
    return (np.mean(means), np.std(means))


def plot():
    """Makes a nice graph with error bars."""
    raise NotImplementedError


def get_iterations(iterations, thermalization):
    """Returns normal iterations and thermalization iterations."""
    normal_iterations = int(iterations * (1 + thermalization))
    t_iterations = int(normal_iterations * (thermalization / (1 + thermalization)))

    return normal_iterations, t_iterations


def get_analytical_param_linspace(values):
    values = values.split(",")
    v_min, v_max, amount = [float(value.strip()) for value in values]

    # create a linspace
    values = np.linspace(v_min, v_max, int(amount) * 100)

    return values


def get_param_linspace(values):
    values = values.split(",")
    v_min, v_max, amount = [float(value.strip()) for value in values]

    # create a linspace
    values = np.linspace(v_min, v_max, int(amount))

    return values


def get_parameters(filename):
    """Loads a parameter file and returns"""

    with open(filename) as f:
        lines = f.readlines()

    # Go through every line and construct
    param_values = []
    param_items = 1

    analytical_param_values = []
    analytical_param_items = 1

    for line in lines:
        p_values = a_values = line.split("=")[1].strip()

        # check if multiple values
        if "," in p_values:
            p_values = get_param_linspace(p_values)
            param_items = len(p_values)

            a_values = get_analytical_param_linspace(a_values)
            analytical_param_items = len(a_values)

        elif "x" in p_values:
            p_values = a_values = [None]
        else:
            p_values = a_values = [float(p_values)]

        param_values.append(p_values)
        analytical_param_values.append(a_values)

    # make sure every value is repeated
    p_parameters = [np.tile(value, param_items) if len(value) < param_items else value for value in param_values]
    a_parameters = [
        np.tile(value, analytical_param_items) if len(value) < analytical_param_items else value
        for value in analytical_param_values
    ]

    p_parameters = [(int(N), J, T, h, J_t, pbc) for N, J, T, h, J_t, pbc in zip(*p_parameters)]
    a_parameters = [(int(N), J, T, h, J_t, pbc) for N, J, T, h, J_t, pbc in zip(*a_parameters)]

    return p_parameters, a_parameters


def get_xlabel(filename):
    with open(filename) as f:
        lines = f.readlines()

    for line in lines:
        if "," in line:
            # get first part
            variable_name = line.split("=")[0].strip().lower()
            p_values = get_param_linspace(line.split("=")[1])
            a_values = get_analytical_param_linspace(line.split("=")[1])

            if variable_name == "n":
                return "N", p_values, a_values
            elif variable_name == "j":
                return "J", p_values, a_values
            elif variable_name == "t":
                return "T", p_values, a_values
            elif variable_name == "h":
                return "h", p_values, a_values
            elif variable_name == "J_t":
                return "J/T", p_values, a_values
            elif variable_name == "pbc":
                return "PBC", p_values, a_values
            else:
                continue

    # no value found
    return "X", [1, 2, 3], [1, 2, 3]


def initialize_system(N):
    """Initializes a random system and returns the even and odd items."""
    system = np.random.choice([1, -1], N)

    return system, system[1::2], system[::2]


def get_free_energy(j, t, n):
    return 2 * j - t * np.log(n - 1)


def get_critical_temperature(j, t, n):
    return (2 * j) / (np.log(n - 1))


def mean_field_approximation(guess, J, T, h, epsilon):
    """Computes the magnetization based on the Mean-Field Approximation."""

    answer = np.tanh((2 * J * guess + h) / T)

    if abs(answer - guess) > epsilon:
        return mean_field_approximation(answer, J, T, h, epsilon)
    else:
        return answer


def get_exact_solution(h, J, T):
    """Get the exact solution to the 1D-Ising model."""

    beta = 1 / T

    m = (np.sinh(beta * h)) / (np.sqrt(np.sinh(beta * h) ** 2 + np.exp(-4 * beta * J)))
    # m = (np.sin(h) * beta * h) / (np.sqrt(np.sin(h ** 2) * beta * h + np.exp(-4 * beta * J)))

    return m
