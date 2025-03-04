import re
import matplotlib.pyplot as plt
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
from ..scripts.data_script import *

def create_fig():
    plt.tight_layout()
    fig = plt.figure(figsize=(5, 5), frameon=False)
    fig.patch.set_visible(False)
    ax = fig.add_axes([0, 0, 1, 1])
    remove_ticks(ax)
    ax.axis('off')
    return fig, ax

def remove_ticks(ax):
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    ax.axis('equal')
    ax.axis('off')

def plot_and_save_data(num_points=5000, radius_1=5, radius_2=10, noise_std=0.4, num_samples=5000, filename='generated_data_plot.png'):

    sampled_df, l0, l1 = generate_sample_pair_wise_data(num_points, radius_1, radius_2, noise_std, num_samples)
    
    fig, ax = create_fig()
    ax.scatter(l0['X'], l0['Y'], color='blue', label='Label 0', alpha=0.6)
    ax.scatter(l1['X'], l1['Y'], color='red', label='Label 1', alpha=0.6)
    ax.legend()
    
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.show()
    
    return sampled_df
def extract_loss_from_log(log_filename):
    """
    Extracts loss values from a log file.

    Args:
        log_filename (str): Path to the log file.

    Returns:
        list: A list of tuples (epoch, loss).
    """
    loss_pattern = re.compile(r"Epoch (\d+), Loss: ([\d\.]+)")
    epochs, losses = [], []

    with open(log_filename, "r") as file:
        for line in file:
            match = loss_pattern.search(line)
            if match:
                epoch = int(match.group(1))
                loss = float(match.group(2))
                epochs.append(epoch)
                losses.append(loss)

    return epochs, losses

def plot_loss_curve(log_filename):
    """
    Reads a log file and plots the training loss.

    Args:
        log_filename (str): Path to the log file.
    """
    epochs, losses = extract_loss_from_log(log_filename)

    if not epochs:
        print("No loss data found in log file.")
        return

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, losses, marker="o", linestyle="-", label="Training Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.legend()
    plt.grid()
    plt.show()

# if __name__ == "__main__":
#     log_path = input("Enter log file path: ")
#     plot_loss_curve(log_path)
