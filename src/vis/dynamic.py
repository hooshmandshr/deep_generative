"""Class for visualizing dynamical system data sets."""


import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D 
from sklearn.decomposition import PCA

import sys
sys.path.append("..")
from dynamics import MarkovDynamicsDiagnostics


class TrajectoryPlot(object):

    def __init__(self, data):
        """Plots the evolution of paths in data.

        params:
        -------
        data: numpy.ndarray
            Shape of data must be (num_paths, num_time_steps, dim)
        three_dim: bool
            If true a 3D plot of the paths is plotted as well.
        """
        self.data = data
        self.num_paths, self.num_time_steps, self.dim = data.shape

        self.pca = PCA(n_components=self.dim)
        self.pca.fit(data.reshape([-1, self.dim]))

    def plot(self, axes=[0], plt_ax=None, PCA=True, data=None, **kwargs):
        """Plots the samples on the specified axes.

        params:
        -------
        axes: list of int
            Length of list should not be greater than 3.
        plt_ax: plt.ax
            Axis for plotting the trajectory.
        PCA: bool
            If True the axis are the PCA basis. Otherwise, the dimensions are
            used.
        """
        fig, ax = None, plt_ax
        data_ = data
        if data is None:
            data_ = self.data

        ax_str = "Dim {}"

        if PCA:
            #Project down to the PCA basis
            data_ = np.matmul(data_, self.pca.components_[axes].T)
            ax_str = "PC {}"
        else:
            data_ = data_[:, :, axes]

        if len(axes) == 1:
            # Plot one axis vs. time.
            if plt_ax is None:
                fig, ax = plt.subplots()

            ax.plot(range(data_.shape[0]), data_[:, :, 0].T, **kwargs)
            ax.set_xlabel("Time step")
            ax.set_ylabel(ax_str.format(axes[0]))

        elif len(axes) == 2:
            # Plot one axis vs. a second one.
            if plt_ax is None:
                fig, ax = plt.subplots()

            ax.plot(data_[:, :, 0].T, data_[:, :, 1].T, **kwargs)
            ax.set_xlabel(ax_str.format(axes[0]))
            ax.set_ylabel(ax_str.format(axes[1]))

        elif len(axes) == 3:
            # Plot 3d trajectories in the plots.
            if plt_ax is None:
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
            for i in range(data_.shape[0]):
                ax.plot(data_[i, :, 0], data_[i, :, 1], data_[i, :, 2],
                        **kwargs)

                ax.set_xlabel(ax_str.format(axes[0] + 1))
                ax.set_ylabel(ax_str.format(axes[1] + 1))
                ax.set_zlabel(ax_str.format(axes[2] + 1))


class Diagnostics(object):

    def __init__(self, dynamics, session, samples, time_forward=2,
            n_examples=2, grid_size=20):
        """

        params:
        -------
        dynamics: dynamics.MarkovDynamics
            The dynamical systems for which the diagnostics is run.
        session: tf.Session
            Tensorflow session under which the dynamics model is trained.
        samples: np.ndarray
            Shape must be (N, T, dim)
        grid_size: np.meshgrid
            Number of points in the 2D grid.
        """
        _, _, self.dim = samples.shape
        self.samples = samples
        all_states = samples.reshape([-1, self.dim])
        self.grid_size = grid_size

        self.dynamics = dynamics
        self.sess = session

        self.mins = all_states.min(axis=0)
        self.maxs = all_states.max(axis=0)
        self.steps = (self.maxs - self.mins) / float(grid_size)
        self.means = all_states.mean(axis=0)

        # Set up extrapolation object
        grid_area = grid_size * grid_size
        self.n_examples = n_examples
        self.time_forward = time_forward
        self.extra = MarkovDynamicsDiagnostics(
            dynamics, n_samples=n_examples,
            grid_size=grid_area, time_forward=time_forward)

    def update_samples(self, samples):
        """Updates sample paths that are plotted along with quiver plots.

        samples: np.ndarray
            Shape must be (N, T, dim)
        """
        if not samples.shape[-1] == self.dim:
            raise ValueError(
                    "Samples shape must be (?, ?, {})".format(self.dim))
        self.samples = samples
        all_states = samples.reshape([-1, self.dim])
        self.mins = all_states.min(axis=0)
        self.maxs = all_states.max(axis=0)
        self.steps = (self.maxs - self.mins) / float(self.grid_size)
        self.means = all_states.mean(axis=0)

    def get_projection_matrix(self, axis1=0, axis2=1, pca=False):
        """Get projection matrix onto the specified axes."""
        proj_mat = np.zeros([2, self.dim])
        proj_mat[:, [axis1, axis2]] = np.eye(2)
        return proj_mat

    def get_extrapolations(self, init_points):
        """Given the object samples get extrapolation from end/strat points.

        params:
        -------
        init_points: np.ndarray
            Initial points to extrapolate from.
        """
        return self.extra.run_extrapolate(
                session=self.sess, states=init_points)

    def get_grid_values(self, axis1=0, axis2=1, pca=False, n_path=5, ax=None):
        """Gets quiver plots for axis1 and axis2.

        params:
        -------
        axis1: int
            First axis on which the path is plotted.
        axis2: int
            Second axis on which the path is plotted.
        pca: bool
            If true, the grid and path are projected on the PCA basis space.
        n_path: int
            Number of paths to be plotted on the quiver plot.
        ax: None or plt.ax
        """
        if ax is None:
            fig1, ax = plt.subplots()

        x = np.arange(self.mins[axis1], self.maxs[axis1], self.steps[axis1])
        y = np.arange(self.mins[axis2], self.maxs[axis2], self.steps[axis2])
        u, v = np.meshgrid(x, y)
        grid = np.array([u.ravel(), v.ravel()]).T
        # Project onto the two axes
        proj_mat = self.get_projection_matrix(axis1=axis1, axis2=axis2, pca=pca)
        grid = np.matmul(grid, proj_mat)
        means = self.means + 0.
        means[[axis1, axis2]] = 0.
        grid += means

        out_grid = self.extra.run_extrapolate(session=self.sess, states=grid, name="grid")

        if isinstance(out_grid, tuple):
            out_grid = out_grid[0]

        uv = (out_grid[:, 1, :] - out_grid[:, 0, :])
        uv = np.matmul(uv, proj_mat.T)

        u_ = uv[:, 0].reshape(u.shape)
        v_ = uv[:, 1].reshape(v.shape)
        m_ = np.hypot(u_, v_)

        ax.set_title('Arrows scale with plot width, not view')
        Q = ax.quiver(x, y, u_, v_, m_, units='width')

        #qk = ax1.quiverkey(Q, 0.9, 0.9, 2, r'$2 \frac{m}{s}$', labelpos='E',
        #                   coordinates='figure')
        for i in range(n_path):
            ax.scatter(self.samples[i, 0, axis1],
                        self.samples[i, 0, axis2])
            ax.plot(self.samples[i, :, axis1].ravel(),
                     self.samples[i, :, axis2].ravel())
        plt.axis('equal')

