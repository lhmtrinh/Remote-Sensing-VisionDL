import numpy as np
from scipy.stats import gaussian_kde

class DenseWeight:
    """
    A class to calculate dense weights for labels using Kernel Density Estimation (KDE).
    The dense weights are used to adjust for label density in a dataset.
    """
    def __init__(self, alpha=0.5, epsilon=1e-4):
        """
        Initialize the DenseWeight class.

        Args:
            alpha (float, optional): Scaling factor for the weights. Default is 0.5.
            epsilon (float, optional): Minimum value for weights to avoid zero values. Default is 1e-4.
        """
        self.alpha = alpha
        self.epsilon = epsilon
        self.min = None
        self.max = None
        self.mean_weight = None
        self.kde = None

    def fit(self, labels):
        """
        Fit the KDE to the provided labels and calculate initial weights.

        Args:
            labels (array-like): Array of labels to fit the KDE.
        """
        # Create a gaussian Kernel Density Estimate
        self.kde = gaussian_kde(labels)

        # Evaluate the KDE for each point in labels
        pdf_values = self.kde(labels)
        self.min = np.min(pdf_values)
        self.max = np.max(pdf_values)

        # Calculate initial weights
        weights = self._weight_value(labels)

        # Set mean weight
        self.mean_weight = np.mean(weights)

    def _normalize_kde(self, labels):
        """
        Normalize the KDE values to the range [0, 1].

        Args:
            labels (array-like): Array of labels to normalize the KDE values.

        Returns:
            np.array: Normalized KDE values.
        """
        normalized_pdf = (self.kde(labels) - self.min) / (self.max - self.min)
        return normalized_pdf

    def _weight_value(self, labels):
        """
        Calculate weight values based on normalized KDE values.

        Args:
            labels (array-like): Array of labels to calculate the weight values.

        Returns:
            np.array: Calculated weight values.
        """
        normalized_pdf = self._normalize_kde(labels)
        weighted_values = np.maximum(1 - self.alpha * normalized_pdf, self.epsilon)
        return weighted_values

    def dense_weight(self, labels):
        """
        Calculate dense weights for the provided labels.

        Args:
            labels (array-like): Array of labels to calculate dense weights.

        Returns:
            np.array: Dense weights for the labels.
        """
        weights = self._weight_value(labels)
        dense_weights = weights / self.mean_weight
        return dense_weights