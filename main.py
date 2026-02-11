# Basic imports
import silence_tensorflow.auto
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

# Load and visualize multidimensional data
from visualization import ScatterPlotVisualizer

# Plotting
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib as mpl

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

        # Dataset
        self.samples_probs = None
        self.labels = None
        self.dataset = None

    def set_params(self, cluster_probs, mus, covs):
        """Set user-defined GMM parameters."""

        self.cluster_probs = tf.constant(cluster_probs, dtype=tf.float32)
        self.mus = tf.constant(mus, dtype=tf.float32)
        self.covs = tf.constant(covs, dtype=tf.float32)

        self._validate_params()

    def set_random_params(self, cov_type: str = "diagonal"):
        """Generate and set random GMM parameters."""

        cluster_probs = tf.constant(
            np.random.dirichlet(np.ones(self.n_components)),
            dtype=tf.float32
        )

        mus = tf.constant(
            np.random.rand(self.n_components, self.n_dimensions) * 10 - 5,
            dtype=tf.float32
        )

        covs = tf.constant(
            self._generate_random_covariances(cov_type),
            dtype=tf.float32
        )

        self.set_params(cluster_probs, mus, covs)

    def _generate_random_covariances(self, cov_type: str = "diagonal"):
        """Generate random covariance matrices based on the specified type."""

        covs = []
        eps = 1e-6

        for _ in range(self.n_components):

            if cov_type == "spherical":
                sigma2 = np.random.rand() * 0.5 + 0.1
                cov = (sigma2 + eps) * np.eye(self.n_dimensions)

            elif cov_type == "diagonal":
                diag = np.random.rand(self.n_dimensions) * 0.5 + 0.1
                cov = np.diag(diag)

            elif cov_type == "full":
                A = np.random.randn(self.n_dimensions, self.n_dimensions)
                cov = A @ A.T + eps * np.eye(self.n_dimensions)

            else:
                raise ValueError("cov_type must be 'spherical', 'diagonal', or 'full'")

            covs.append(cov)

        return np.array(covs)

    def _validate_params(self):

        if self.cluster_probs is None or self.mus is None or self.covs is None:
            raise ValueError("GMM parameters are not set. Use set_params() or set_random_params(cov_type).")

        if self.cluster_probs.shape[0] != self.n_components:
            raise ValueError("cluster_probs shape mismatch.")

        if self.mus.shape != (self.n_components, self.n_dimensions):
            raise ValueError("mus shape mismatch.")

        if self.covs.shape != (self.n_components, self.n_dimensions, self.n_dimensions):
            raise ValueError("covs shape mismatch.")

    def generate_data(self, include_mus: bool = True):
        """
        Generate a synthetic data sampled from a GMM.
        Args:
            random_params (bool): Whether to generate random GMM parameters.
            cov_type (str): If random_params is True, user must specify the type of covariance matrices ("spherical", "diagonal", "full").
            include_mus (bool): Whether to include the original means in the dataset.
        Returns:
            dataset (tf.Tensor): Generated synthetic dataset.
        """

        # (1) Validate loaded parameters
        self._validate_params()

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

# Example usage
if __name__ == "__main__":
    generator = SyntheticGMMGenerator(
        n_samples=100,
        n_dimensions=4,
        n_components=10,
        seed=42
    )

    # Data generation
    generator.set_random_params(cov_type="full") # must be randomly set or loaded by the user (set_params)
    data = generator.generate_data(include_mus=True) # generate random parameters with diagonal covariance matrices
    mus = generator.mus
    covs = generator.covs
    cluster_probs = generator.cluster_probs
    print("Means:\n", mus.numpy())
    print("Covs:\n", covs.numpy())
    print("Cluster probs:\n", cluster_probs.numpy())

    # Visualization
    visualizer = ScatterPlotVisualizer(
        data=data,
        mus=mus,
        covs=covs,
        cluster_probs=cluster_probs
    )
    visualizer.plot_pca(samples_size=20, figsize=(8, 6))
    visualizer.plot_splom(samples_size=20, figsize=(12, 12))