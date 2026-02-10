# Basic imports
import silence_tensorflow.auto
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

# Load and visualize multidimensional data
import pandas as pd
import seaborn as sns
from sklearn.manifold import TSNE

# Plotting
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm

# Synthetic Data Generator with Gaussian Mixture Models (GMM) using Tensorflow Probability

tfd = tfp.distributions

class SyntheticGMMGenerator:
    def __init__(
        self,
        n_samples: int,
        n_dimensions: int,
        n_components: int,
        seed: int | None = None
    ):
        self.n_samples = n_samples
        self.n_dimensions = n_dimensions
        self.n_components = n_components
        self.seed = seed

        if seed is not None:
            tf.random.set_seed(seed)
            np.random.seed(seed)

        # Mixture parameters
        self.cluster_probs = None
        self.mus = None
        self.covs = None

        # Samples
        self.samples_probs = None
        self.labels = None

        # Dataset
        self.dataset = None

    def plot_gmm(self, mus, covs, cluster_probs, ax=None, n_std=2):
        """
        Plot GMM parameters.
        
        Parameters
        ----------
        mus : array (K, 2)
            Means of the Gaussians
        covs : array (K, 2, 2)
            Covariance matrices
        cluster_probs : array (K,)
            Mixture weights (used as alpha)
        n_std : int
            Number of standard deviations for ellipse radius
        """

        if ax is None:
            fig, ax = plt.subplots(nrows=self.n_dimensions, ncols=self.n_dimensions,figsize=(6, 6), sharex=True)

        colors = cm.gist_rainbow(np.linspace(0, 1, self.n_components))

        # Plot gaussian curves for each dimension
        for dim_idx in range(self.n_dimensions):
            for clust_idx in range(mus.shape[0]):
                mu = mus[clust_idx, dim_idx]
                sigma = np.sqrt(covs[clust_idx][dim_idx, dim_idx])
                alpha = cluster_probs[clust_idx]
                color = colors[clust_idx]

                x = np.linspace(mu - 4*sigma, mu + 4*sigma, 100)
                y = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

                ax[dim_idx, dim_idx].plot(x, y, color=color, alpha=float(alpha))
                ax[dim_idx, dim_idx].grid(True, which='both', axis='x', linestyle='--', alpha=0.5)
                ax[dim_idx, dim_idx].set_yticks([])  # Hide y-axis ticks for clarity
                # ax[dim_idx, dim_idx].set_ylim(0, None)  # Set y-axis limit to start at 0

        # Plot ellipses for each pair of dimensions
        for i in range(self.n_dimensions):
            for j in range(i+1, self.n_dimensions):
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

        return ax

    def params_settings(self, plot_params=False):
        """Print the parameters of the GMM."""

        cluster_probs = self.cluster_probs.numpy() if self.cluster_probs is not None else None
        mus = self.mus.numpy() if self.mus is not None else None
        covs = self.covs.numpy() if self.covs is not None else None

        print("Cluster weights (probs):", cluster_probs)
        print("Means (mus):")
        print("Covariances (covs):", covs)

        self.ax = self.plot_gmm(mus=mus, covs=covs, cluster_probs=cluster_probs)

        if plot_params:
            plt.show()

    def load_params(self, cluster_probs, mus, covs):
        """Load GMM parameters from user input."""
        self.cluster_probs = tf.constant(cluster_probs, dtype=tf.float32)
        self.mus = tf.constant(mus, dtype=tf.float32)
        self.covs = tf.constant(covs, dtype=tf.float32)

    def load_random_params(self, cov_type: str = "diagonal"):
        """Generate random GMM parameters."""
        # Cluster weights
        self.cluster_probs = tf.constant(
            np.random.dirichlet(np.ones(self.n_components)),
            dtype=tf.float32
        )

        # Means
        self.mus = tf.constant(
            np.random.rand(self.n_components, self.n_dimensions) * 10 - 5,
            dtype=tf.float32
        )

        if cov_type == "spherical":
            # Spherical covariance matrices
            covs = []
            eps = 1e-6  # numerical stability
            for _ in range(self.n_components):
                # Single variance per cluster
                sigma2 = np.random.rand() * 0.5 + 0.1   # scalar variance
                cov = (sigma2 + eps) * np.eye(self.n_dimensions)
                covs.append(cov)
            self.covs = tf.constant(np.array(covs), dtype=tf.float32)

        elif cov_type == "diagonal":
            # Diagonal covariance matrices
            covs = []
            for _ in range(self.n_components):
                diag = np.random.rand(self.n_dimensions) * 0.5 + 0.1
                covs.append(np.diag(diag))
            self.covs = tf.constant(covs, dtype=tf.float32)

        elif cov_type == "full":
        # Full covariance matrices
            covs = []
            eps = 1e-3  # numerical stability
            for _ in range(self.n_components):
                A = np.random.randn(self.n_dimensions, self.n_dimensions)
                cov = A @ A.T + eps * np.eye(self.n_dimensions)
                covs.append(cov)
            self.covs = tf.constant(np.array(covs), dtype=tf.float32)

    def generate_data(self, random_params: bool = True, cov_type: str = "diagonal", include_mus: bool = True):
        """
        Generate a synthetic data sampled from a GMM.
        Args:
            random_params (bool): Whether to generate random GMM parameters.
            cov_type (str): If random_params is True, user must specify the type of covariance matrices ("spherical", "diagonal", "full").
            include_mus (bool): Whether to include the original means in the dataset.
        Returns:
            dataset (tf.Tensor): Generated synthetic dataset.
        """

        # (1) Load parameters
        if random_params:
            self.load_random_params(cov_type=cov_type)
        
        else:
            if self.cluster_probs is None or self.mus is None or self.covs is None:
                raise ValueError("You must load parameters using load_params() before generating samples.")

        scale_tril = tf.linalg.cholesky(self.covs) # Cholesky decomposition -> Input for MultivariateNormalTriL

        # (2) GMM definition
        cat = tfd.Categorical(probs=self.cluster_probs)
        normals = tfd.MultivariateNormalTriL(
            loc=self.mus,
            scale_tril=scale_tril
        )

        gmm = tfd.MixtureSameFamily(
            mixture_distribution=cat,
            components_distribution=normals
        ) # joint distribution of the GMM

        # (3) Sample from the GMM -> Synthetic data generation
        if include_mus:
            self.dataset = gmm.sample(self.n_samples-self.n_components, seed=self.seed)
            self.dataset = tf.concat([self.dataset, self.mus], axis=0) # combine with the original mus, which are considered as the typical samples of the copycat dataset

        else:
            self.dataset = gmm.sample(self.n_samples, seed=self.seed)

        # # Sample component indices
        # labels = cat.sample(sample_shape=self.n_samples, seed=self.seed)

        # # Sample from each Gaussian component
        # samples = normals.sample(seed=self.seed)

        # # Gather samples according to labels
        # samples = tf.gather(samples, labels)

        # self.dataset = samples.numpy()
        # self.labels = labels.numpy()

        return self.dataset

    def plot_tsne(
        self,
        samples_size: int = 20,
        perplexity: int = 30,
        random_state: int = 42,
        figsize=(8, 6)
    ):
        """
        Plot the dataset using t-SNE.
        """

        if self.dataset is None:
            raise RuntimeError("You must call generate_data() before plotting.")

        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            random_state=random_state
        )

        if self.dataset.shape[1] == 2:
            data_2d = self.dataset
            print(self.dataset)
        else:
            data_2d = tsne.fit_transform(self.dataset)

        # cmap = cm.get_cmap(colormap, self.n_components)

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

        plt.xlabel("t-SNE 1")
        plt.ylabel("t-SNE 2")
        plt.legend()
        plt.title("Synthetic GMM dataset (t-SNE projection)")
        plt.tight_layout()
        plt.show()

    def plot_dataset(self, samples_size):
        self.ax = self.plot_gmm(mus=self.mus, covs=self.covs, cluster_probs=self.cluster_probs)
        plt.show()

# Example usage
if __name__ == "__main__":
    generator = SyntheticGMMGenerator(
        n_samples=100,
        n_dimensions=4,
        n_components=10,
        seed=42
    )

    # generator.load_params(cluster_probs, mus, covs) # must be set if random_params=False
    data = generator.generate_data(random_params=True, cov_type="diagonal") # generate random parameters with diagonal covariance matrices
    # generator.params_settings(plot_params=False)
    generator.plot_dataset(samples_size=20)
    generator.plot_tsne(samples_size=20, perplexity=30) # perplexity must be < n_samples