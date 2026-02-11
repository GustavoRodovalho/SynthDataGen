import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib as mpl
import numpy as np
from sklearn.decomposition import PCA

class ScatterPlotVisualizer():
    def __init__(
            self,
            data: tf.Tensor = None,
            mus: tf.Tensor = None,
            covs: tf.Tensor = None,
            cluster_probs: tf.Tensor = None,
            colormap: str = "gist_rainbow"):
        self.data = data
        self.mus = mus
        self.covs = covs
        self.cluster_probs = cluster_probs
        self.colormap = colormap

    def plot_pca(
        self,
        samples_size: int = 30,
        mus_size: int = 60,
        random_state: int = None,
        figsize=(8, 6)
    ):
        """
        Plot the two principal components of the data using PCA.
        """

        if self.data is None:
            raise RuntimeError("You must specify generate_data() to obtain the data.")

        pca = PCA(
            n_components=2,
            random_state=random_state
        )

        if self.data.shape[1] == 2:
            data_2d = self.data
            mus_2d = self.mus
        else:
            data_2d = pca.fit_transform(self.data)
            mus_2d = pca.transform(self.mus)

        colors = mpl.colormaps[self.colormap](np.linspace(0, 1, self.mus.shape[0]))
        colors[:, 3] = self.cluster_probs.numpy()

        plt.figure(figsize=figsize)

        plt.scatter(
            data_2d[:, 0],
            data_2d[:, 1],
            s=samples_size,
            color='white',
            edgecolors='k',
            label="Samples",
            alpha=0.8
        )

        plt.scatter(
            mus_2d[:, 0],
            mus_2d[:, 1],
            s=mus_size,
            color=colors,
            edgecolors='k',
            label="Means"
        )

        plt.xlabel("PC 1")
        plt.ylabel("PC 2")
        plt.legend()
        plt.title("Synthetic GMM data in PCA space")
        plt.tight_layout()
        plt.show()

    def plot_splom(
        self,
        n_std=2,
        samples_size: int = 30,
        figsize=(10, 10)
    ):
        """
        Plot GMM parameters.
        """

        mus = self.mus.numpy()
        covs = self.covs.numpy()
        cluster_probs = self.cluster_probs.numpy()

        fig, ax = plt.subplots(nrows=mus.shape[1], ncols=mus.shape[1],figsize=figsize, sharex=True)

        cmap = plt.get_cmap(self.colormap)
        colors = cmap(np.linspace(0, 1, self.mus.shape[0]))

        if self.data is None:
            raise RuntimeError("You must specify generate_data() to obtain the data.")
        else:
            if self.data is not None:
                # Plot data points
                for j in range(self.data.shape[1]):
                    for i in range(j+1, self.data.shape[1]):
                        ax[i, j].scatter(
                            self.data[:, j],
                        self.data[:, i],
                        s=samples_size,
                        color='white',
                        edgecolors='k',
                        alpha=0.8
                    )

        # Plot gaussian curves for each dimension
        for dim_idx in range(mus.shape[1]): # loop over dimensions
            for clust_idx in range(mus.shape[0]): # loop over clusters
                mu = mus[clust_idx, dim_idx]
                sigma = np.sqrt(covs[clust_idx][dim_idx, dim_idx])
                alpha = cluster_probs[clust_idx]
                color = colors[clust_idx]

                x = np.linspace(mu - 4*sigma, mu + 4*sigma, 100)
                y = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

                ax[dim_idx, dim_idx].plot(x, y, color=color, alpha=float(alpha))
                ax[dim_idx, dim_idx].grid(True, which='both', axis='x', linestyle='--', alpha=0.5)
                ax[dim_idx, dim_idx].set_yticks([])  # Hide y-axis ticks for clarity

        # Plot ellipses for each pair of dimensions
        for i in range(mus.shape[1]):
            for j in range(i+1, mus.shape[1]):
                for clust_idx in range(mus.shape[0]):
                    mu = mus[clust_idx, [i, j]]
                    cov = covs[clust_idx][np.ix_([i, j],[i, j])]
                    alpha = cluster_probs[clust_idx]
                    color = colors[clust_idx]

                    # Eigen decomposition of covariance
                    eigvals, eigvecs = np.linalg.eigh(cov)

                    # Sort eigenvalues (largest first)
                    order = eigvals.argsort()[::-1]
                    eigvals, eigvecs = eigvals[order], eigvecs[:, order]

                    # Compute ellipse angle
                    angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))

                    # Width and height (n_std confidence)
                    width, height = 2 * n_std * np.sqrt(eigvals)

                    ellipse = mpatches.Ellipse(
                        xy=mu,
                        width=width,
                        height=height,
                        angle=angle,
                        facecolor=color,
                        edgecolor=color,
                        alpha=float(alpha),
                        linewidth=2
                    )

                    ax[j, i].add_patch(ellipse)
                    ax[j, i].scatter(mu[0], mu[1], color='k', s=50, marker='x')
                    ax[j, i].grid(True)

        plt.show()