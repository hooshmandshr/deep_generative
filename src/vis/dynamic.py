"""Class for visualizing dynamical system data sets."""


import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D 
from sklearn.decomposition import PCA


def plot_paths(data, three_dim=False, **kwargs):
    """Plots the evolution of paths in data.

    params:
    -------
    data: numpy.ndarray
        Shape of data must be (num_paths, num_time_steps, dim)
    three_dim: bool
        If true a 3D plot of the paths is plotted as well.
    """
    num_paths, num_time_steps, dim = data.shape
    if dim > 3:
        pca = PCA(n_components=3)
        data = pca.fit_transform(data.reshape([-1, dim])).reshape(
                [num_paths, num_time_steps, 3])
    new_dim = data.shape[-1]

    if three_dim:
        if not new_dim == 3:
            raise ValueError("Data dimensionality is less that 3")
        # Add a three dimensional plot of the evolution too.
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for i in range(num_paths):
            ax.plot(data[i, :, 0], data[i, :, 1], data[i, :, 2])
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_zlabel("PC3")

    else:
        fig, ax = plt.subplots(1, new_dim + 1)
        for i in range(new_dim):
            ax[i].plot(data[:, :, i].T, **kwargs)
            ax[i].set_xlabel("time")
            ax[i].set_ylabel("PC{}".format(i + 1))
        ax[-1].plot(data[:, :, 0].T, data[:, :, 1].T, **kwargs)
        ax[-1].set_ylabel("PC2")
        ax[-1].set_xlabel("PC1")
 
