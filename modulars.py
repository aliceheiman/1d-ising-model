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


def get_energy_density(system, pbc, J_t, h, N):
    """Computes the energy density of a certain system."""
    system_shifted = np.hstack((system[1:], pbc * system[0]))

    return (-J_t * np.sum(system * system_shifted) - h * np.sum(system)) / N


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
    all_values = []
    items = 1

    for line in lines:
        values = line.split("=")[1].strip()

        # check if multiple values
        if "," in values:
            values = get_param_linspace(values)

            # save the length
            items = len(values)
        elif "x" in values:
            values = [None]
        else:
            values = [float(values)]

        all_values.append(values)

    # make sure every value is repeated
    parameters = [np.tile(value, items) if len(value) < items else value for value in all_values]

    parameters = [(int(N), J, T, h, J_t, pbc) for N, J, T, h, J_t, pbc in zip(*parameters)]

    return parameters


def get_xlabel(filename):
    with open(filename) as f:
        lines = f.readlines()

    for line in lines:
        if "," in line:
            # get first part
            variable_name = line.split("=")[0].strip().lower()
            values = get_param_linspace(line.split("=")[1])

            if variable_name == "n":
                return "System Size", values
            elif variable_name == "j":
                return "Interspin Exchange Coupling", values
            elif variable_name == "t":
                return "Temperature", values
            elif variable_name == "h":
                return "External Magnetic Field", values
            elif variable_name == "J_t":
                return "Interspin Exchange Coupling over Temperature", values
            elif variable_name == "pbc":
                return "Periodic Boundry Condition", values
            else:
                continue

    # no value found
    return "X", [1, 2, 3]


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

    m = (np.sin(h) * beta * h) / (np.sqrt(np.sin(h ** 2) * beta * h + np.exp(-4 * beta * J)))

    return m
