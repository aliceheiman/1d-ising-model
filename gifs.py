from modulars import *
import numpy as np
import sys
import time
import matplotlib.pyplot as plt
import matplotlib as mpl
from PIL import Image, ImageDraw
import os
import imageio

plt.style.use("seaborn")
sys.setrecursionlimit(100000)


def generate_gif(directory, filename):
    images = []
    for image_name in sorted(os.listdir(directory)):
        images.append(imageio.imread(os.path.join(directory, image_name)))
    imageio.mimsave(filename, images)


experiment = "experiment9f"
systems = np.load(f"data/systems/{experiment}.npy", allow_pickle=True)

fig_name = f"_figures/gifs/{experiment}.gif"  # false
# fig_name = False

size = 50

data = np.load(f"data/{experiment}.npy", allow_pickle=True)
data = data.item()

graph_params1 = [
    ("N", data["params"]["N"]),
    ("\lambda", data["params"]["pbc"]),
]
