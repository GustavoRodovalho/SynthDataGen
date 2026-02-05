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

import silence_tensorflow.auto
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.manifold import TSNE

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

        self.cluster_probs = None
        self.mus = None
        self.covs = None
        self.labels = None
        self.dataset = None

    def generate(self):
        """Generate a synthetic dataset sampled from a GMM."""

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

        # Diagonal covariance matrices
        covs = []
        for _ in range(self.n_components):
            diag = np.random.rand(self.n_dimensions) * 0.5 + 0.1
            covs.append(np.diag(diag))

        self.covs = tf.constant(covs, dtype=tf.float32)
        scale_tril = tf.linalg.cholesky(self.covs)

        # GMM definition
        cat = tfd.Categorical(probs=self.cluster_probs)
        normals = tfd.MultivariateNormalTriL(
            loc=self.mus,
            scale_tril=scale_tril
        )

        gmm = tfd.MixtureSameFamily(
            mixture_distribution=cat,
            components_distribution=normals
        )

        # Sample component indices
        labels = cat.sample(sample_shape=self.n_samples, seed=self.seed)

        # Sample from each Gaussian component
        samples = normals.sample(seed=self.seed)

        # Gather samples according to labels
        samples = tf.gather(samples, labels)

        self.dataset = samples.numpy()
        self.labels = labels.numpy()

        return self.dataset, self.labels

    def plot(
        self,
        colormap: str = "gist_rainbow",
        perplexity: int = 30,
        random_state: int = 42,
        figsize=(8, 6)
    ):
        """
        Plot the dataset using t-SNE and a user-defined colormap.
        """

        if self.dataset is None or self.labels is None:
            raise RuntimeError("You must call generate() before plotting.")

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

        cmap = cm.get_cmap(colormap, self.n_components)

        plt.figure(figsize=figsize)

        for k in range(self.n_components):
            mask = self.labels == k
            plt.scatter(
                data_2d[mask, 0],
                data_2d[mask, 1],
                s=20,
                color=cmap(k),
                label=f"Cluster {k}",
                alpha=0.8
            )

        plt.xlabel("t-SNE 1")
        plt.ylabel("t-SNE 2")
        plt.legend()
        plt.title("Synthetic GMM dataset (t-SNE projection)")
        plt.tight_layout()
        plt.show()

# Example usage
if __name__ == "__main__":
    generator = SyntheticGMMGenerator(
        n_samples=30,
        n_dimensions=2,
        n_components=3,
        seed=42
    )

    data, labels = generator.generate()
    generator.plot(perplexity=40)