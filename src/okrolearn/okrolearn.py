import numpy as np
import pickle
from typing import Tuple, Union, Callable, Optional, List
from scipy import sparse
import pandas as pd
import matplotlib.pyplot as plt
import pstats
import io
import cProfile

class Tensor:
    def __init__(self, data, requires_grad=True):
        self.data = np.array(data)  # Tensor is linked to a numpy array defined as (data)
        self.grad = None  # Gradient on tensor creation is set to none
        self.requires_grad = requires_grad
        self.backward_fn = None  # Backward function set to none

    def __add__(self, other):
        out = Tensor(self.data + other.data)  # Output tensor is the sum of this tensor and the other tensor

        def backward_fn(grad):
            if self.requires_grad:
                self.grad = grad if self.grad is None else self.grad + grad
            if other.requires_grad:
                other.grad = grad if other.grad is None else other.grad + grad

        out.backward_fn = backward_fn  # The output's backward function is defined here
        return out

    def __mul__(self, other):
        out_data = self.data * other.data if isinstance(other, Tensor) else self.data * other
        out = Tensor(out_data)

        def backward_fn(grad):
            if isinstance(other, Tensor):
                if self.requires_grad:
                    self.grad = grad * other.data if self.grad is None else self.grad + grad * other.data
                if other.requires_grad:
                    other.grad = grad * self.data if other.grad is None else other.grad + grad * self.data
            else:
                if self.requires_grad:
                    self.grad = grad * other if self.grad is None else self.grad + grad * other

        out.backward_fn = backward_fn
        return out

    def __rmul__(self, other):
        return self.__mul__(other)  # Handles multiplication when the tensor is on the right

    def apply(self, func):
        vectorized_func = np.vectorize(func)
        out = Tensor(vectorized_func(self.data))

        def backward_fn(grad):
            if self.requires_grad:
                self.grad = grad * vectorized_func(self.data,
                                                   derivative=True) if self.grad is None else self.grad + grad * vectorized_func(
                    self.data, derivative=True)

        out.backward_fn = backward_fn
        return out

    def dot(self, other):
        out = Tensor(np.dot(self.data, other.data))

        def backward_fn(grad):
            if self.requires_grad:
                self.grad = np.dot(grad, other.data.T) if self.grad is None else self.grad + np.dot(grad, other.data.T)
            if other.requires_grad:
                other.grad = np.dot(self.data.T, grad) if other.grad is None else other.grad + np.dot(self.data.T, grad)

        out.backward_fn = backward_fn
        return out

    def to_numpy(self):
        return self.data

    def to_sparse(self, format='csr'):
        """
        Convert the tensor to a sparse representation.

        Parameters:
        - format: The sparse matrix format. Options are 'csr', 'csc', 'coo', 'lil', 'dok', 'bsr'.
                  Default is 'csr' (Compressed Sparse Row).

        Returns:
        - A scipy sparse matrix in the specified format.
        """
        if not isinstance(self.data, np.ndarray):
            raise ValueError("Tensor data must be a numpy array")

        if self.data.ndim != 2:
            raise ValueError("Only 2D tensors can be converted to sparse format")

        if format not in ['csr', 'csc', 'coo', 'lil', 'dok', 'bsr']:
            raise ValueError("Unsupported sparse format")

        # Convert to the specified sparse format
        sparse_matrix = getattr(sparse, format + '_matrix')(self.data)

        return sparse_matrix

    @staticmethod
    def from_sparse(sparse_matrix):
        """
        Convert a sparse matrix to a Tensor.

        Parameters:
        - sparse_matrix: A scipy sparse matrix.

        Returns:
        - A Tensor object.
        """
        if not sparse.issparse(sparse_matrix):
            raise ValueError("Input must be a scipy sparse matrix")

        # Convert sparse matrix to dense numpy array
        dense_array = sparse_matrix.toarray()

        # Create and return a new Tensor
        return Tensor(dense_array)

    def transpose(self):
        out = Tensor(self.data.T)

        def backward_fn(grad):
            if self.requires_grad:
                self.grad = grad.T if self.grad is None else self.grad + grad.T

        out.backward_fn = backward_fn
        return out

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], tuple):
            shape = shape[0]

        reshaped_data = self.data.reshape(shape)
        reshaped_tensor = Tensor(reshaped_data)

        def backward_fn(grad):
            if self.requires_grad:
                self.grad = grad.reshape(self.data.shape) if self.grad is None else self.grad + grad.reshape(
                    self.data.shape)

        reshaped_tensor.backward_fn = backward_fn
        return reshaped_tensor

    def batch_norm(self, eps=1e-5):
        mean = np.mean(self.data, axis=0)
        variance = np.var(self.data, axis=0)
        normalized_data = (self.data - mean) / np.sqrt(variance + eps)
        out = Tensor(normalized_data)

        def backward_fn(grad):
            if self.requires_grad:
                self.grad = grad / np.sqrt(variance + eps) if self.grad is None else self.grad + grad / np.sqrt(
                    variance + eps)

        out.backward_fn = backward_fn
        return out

    def swap(self, dim0, dim1):
        axes = list(range(self.data.ndim))
        axes[dim0], axes[dim1] = axes[dim1], axes[dim0]
        swapped_data = np.transpose(self.data, axes)
        out = Tensor(swapped_data)

        def backward_fn(grad):
            if self.requires_grad:
                self.grad = np.transpose(grad, axes) if self.grad is None else self.grad + np.transpose(grad, axes)

        out.backward_fn = backward_fn
        return out

    def polynomial_features(self, degree):
        """
        Generate polynomial features of the given degree for the tensor.
        """
        from itertools import combinations_with_replacement

        n_samples, n_features = self.data.shape
        combs = [combinations_with_replacement(range(n_features), d) for d in range(degree + 1)]
        combs = [item for sublist in combs for item in sublist]

        new_data = np.hstack([np.prod(self.data[:, comb], axis=1, keepdims=True) for comb in combs])
        out = Tensor(new_data)

        def backward_fn(grad):
            if self.requires_grad:
                grad_features = np.zeros_like(self.data)
                for i, comb in enumerate(combs):
                    partial_grad = grad[:, i][:, None]
                    for feature_idx in comb:
                        partial_grad *= self.data[:, feature_idx][:, None]
                    grad_features[:, comb] += partial_grad

                if self.grad is None:
                    self.grad = grad_features
                else:
                    self.grad += grad_features

        out.backward_fn = backward_fn
        return out

    def min_max_scale(self, feature_range=(0, 1)):
        """
        Scale features to a given range (similar to MinMaxScaler in scikit-learn)
        """
        min_val = np.min(self.data, axis=0)
        max_val = np.max(self.data, axis=0)
        scaled_data = (self.data - min_val) / (max_val - min_val)
        scaled_data = scaled_data * (feature_range[1] - feature_range[0]) + feature_range[0]
        out = Tensor(scaled_data)

        def backward_fn(grad):
            if self.requires_grad:
                scale = (max_val - min_val) / (feature_range[1] - feature_range[0])
                if self.grad is None:
                    self.grad = grad * scale
                else:
                    self.grad += grad * scale

        out.backward_fn = backward_fn
        return out

    def truncated_svd(self, n_components):
        """
        Perform Truncated SVD (similar to TruncatedSVD in scikit-learn)
        """
        U, s, Vt = np.linalg.svd(self.data, full_matrices=False)
        U_truncated = U[:, :n_components]
        s_truncated = s[:n_components]
        Vt_truncated = Vt[:n_components, :]

        transformed_data = U_truncated * s_truncated
        out = Tensor(transformed_data)

        def backward_fn(grad):
            if self.requires_grad:
                grad_full = np.dot(grad, Vt_truncated)
                if self.grad is None:
                    self.grad = grad_full
                else:
                    self.grad += grad_full

        out.backward_fn = backward_fn
        return out, Tensor(Vt_truncated)

    def svm(self, y, learning_rate=0.001, num_iterations=5000, C=1.0):
        """
        Basic Support Vector Machine implementation
        """
        if not isinstance(y, Tensor):
            y = Tensor(y)

        m, n = self.data.shape
        w = np.zeros(n)
        b = 0

        for _ in range(num_iterations):
            for i in range(m):
                condition = y.data[i] * (np.dot(self.data[i], w) + b) >= 1
                if condition:
                    w = w - learning_rate * (2 * (1 / num_iterations) * w)
                else:
                    w = w + learning_rate * (y.data[i] * self.data[i] - 2 * (1 / num_iterations) * w)
                    b = b + learning_rate * y.data[i]

        out = Tensor(w)

        def backward_fn(grad):
            if self.requires_grad:
                self.grad = grad * self.data if self.grad is None else self.grad + grad * self.data

        out.backward_fn = backward_fn
        return out, b

    def logistic_regression(self, y, learning_rate=0.01, num_iterations=1000):
        """
        Logistic Regression implementation
        """
        if not isinstance(y, Tensor):
            y = Tensor(y)

        m, n = self.data.shape
        w = np.zeros(n)
        b = 0

        def sigmoid(z):
            return 1 / (1 + np.exp(-z))

        for _ in range(num_iterations):
            z = np.dot(self.data, w) + b
            h = sigmoid(z)
            gradient_w = np.dot(self.data.T, (h - y.data)) / m
            gradient_b = np.sum(h - y.data) / m

            w -= learning_rate * gradient_w
            b -= learning_rate * gradient_b

        out = Tensor(w)

        def backward_fn(grad):
            if self.requires_grad:
                self.grad = grad * self.data if self.grad is None else self.grad + grad * self.data

        out.backward_fn = backward_fn
        return out, b

    def random_forest(self, y, n_trees=10, max_depth=None, min_samples_split=2):
        """
        Random Forest implementation
        """
        if not isinstance(y, Tensor):
            y = Tensor(y)

        class DecisionTree:
            def __init__(self, max_depth=None, min_samples_split=2):
                self.max_depth = max_depth
                self.min_samples_split = min_samples_split
                self.tree = None
                self.classes = np.unique(y.data)

            def build_tree(self, X, y, depth=0):
                n_samples, n_features = X.shape
                n_labels = len(np.unique(y))

                # Handle the case where we have no samples
                if n_samples == 0:
                    return self.classes[0]  # Return the first class as default

                if (depth == self.max_depth or
                        n_samples < self.min_samples_split or
                        n_labels == 1):
                    return np.bincount(y, minlength=len(self.classes)).argmax()

                feature_idx = np.random.randint(n_features)
                threshold = np.median(X[:, feature_idx])

                left_mask = X[:, feature_idx] <= threshold
                right_mask = ~left_mask

                left = self.build_tree(X[left_mask], y[left_mask], depth + 1)
                right = self.build_tree(X[right_mask], y[right_mask], depth + 1)

                return (feature_idx, threshold, left, right)

            def fit(self, X, y):
                self.tree = self.build_tree(X, y)

            def predict_sample(self, x, tree):
                if isinstance(tree, (np.int64, np.int32, int)):
                    return self.classes[tree]
                feature_idx, threshold, left, right = tree
                if x[feature_idx] <= threshold:
                    return self.predict_sample(x, left)
                else:
                    return self.predict_sample(x, right)

            def predict(self, X):
                return np.array([self.predict_sample(x, self.tree) for x in X])

        forest = []
        for _ in range(n_trees):
            tree = DecisionTree(max_depth, min_samples_split)
            indices = np.random.choice(len(self.data), len(self.data), replace=True)
            X_sample, y_sample = self.data[indices], y.data[indices]
            tree.fit(X_sample, y_sample)
            forest.append(tree)

        def predict(X):
            predictions = np.array([tree.predict(X) for tree in forest])
            return np.apply_along_axis(lambda x: np.bincount(x, minlength=len(forest[0].classes)).argmax(), axis=0,
                                       arr=predictions)

        out = Tensor(predict(self.data))

        def backward_fn(grad):
            if self.requires_grad:
                self.grad = np.zeros_like(self.data) if self.grad is None else self.grad

        out.backward_fn = backward_fn
        return out, forest

    @classmethod
    def from_pandas(cls, df):
        """
        Create a Tensor from a pandas DataFrame.

        Parameters:
        - df: pandas DataFrame

        Returns:
        - Tensor object
        """
        return cls(df.values)

    def to_pandas(self, columns=None):
        """
        Convert the Tensor to a pandas DataFrame.

        Parameters:
        - columns: list of column names (optional)

        Returns:
        - pandas DataFrame
        """
        if columns is None:
            return pd.DataFrame(self.data)
        else:
            if len(columns) != self.data.shape[1]:
                raise ValueError("Number of column names must match the number of columns in the data")
            return pd.DataFrame(self.data, columns=columns)

    def split(self, indices_or_sections, axis=0):
        split_data = np.split(self.data, indices_or_sections, axis)
        split_tensors = [Tensor(sd) for sd in split_data]

        def backward_fn(grad, split_index):
            if self.grad is None:
                self.grad = np.zeros_like(self.data)

            if isinstance(indices_or_sections, int):
                split_size = self.data.shape[axis] // indices_or_sections
                start = split_index * split_size
                end = start + split_size
            else:
                start = 0 if split_index == 0 else indices_or_sections[split_index - 1]
                end = indices_or_sections[split_index] if split_index < len(indices_or_sections) else None

            if axis == 0:
                self.grad[start:end] += grad
            elif axis == 1:
                self.grad[:, start:end] += grad
            else:
                slice_obj = [slice(None)] * self.grad.ndim
                slice_obj[axis] = slice(start, end)
                self.grad[tuple(slice_obj)] += grad

        for i, t in enumerate(split_tensors):
            t.backward_fn = lambda grad, i: backward_fn(grad, i)

        return split_tensors

    def standardize(self):
        """
        Standardize the tensor (zero mean, unit variance)
        Similar to scikit-learn's StandardScaler
        """
        mean = np.mean(self.data, axis=0)
        std = np.std(self.data, axis=0)
        standardized_data = (self.data - mean) / (std + 1e-8)  # Add small epsilon to avoid division by zero
        out = Tensor(standardized_data, requires_grad=True)

        def backward_fn(grad):
            if self.requires_grad:
                if self.grad is None:
                    self.grad = grad / (std + 1e-8)
                else:
                    self.grad += grad / (std + 1e-8)

        out.backward_fn = backward_fn
        return out

    def normalize(self):
        """
        Normalize the tensor (zero mean, unit variance)
        Similar to scikit-learn's StandardScaler
        """
        mean = np.mean(self.data, axis=0)
        std = np.std(self.data, axis=0)
        normalized_data = (self.data - mean) / (std + 1e-8)
        out = Tensor(normalized_data, requires_grad=True)

        def backward_fn(grad):
            if self.requires_grad:
                if self.grad is None:
                    self.grad = grad / (std + 1e-8)
                else:
                    self.grad += grad / (std + 1e-8)

        out.backward_fn = backward_fn
        return out


    def cross_val_score(self, model_fn, X, y, cv=5):
        """
        Perform cross-validation and return the scores.

        Parameters:
        - model_fn: A function that takes training data and labels, and returns a fitted model.
                    The model should also have a method `predict` that takes test data and returns predictions.
        - X: Input data, assumed to be a 2D numpy array.
        - y: Labels, assumed to be a 1D numpy array.
        - cv: Number of cross-validation folds (default is 5).

        Returns:
        - scores: A list of scores for each fold.
        """
        n_samples = len(X)
        indices = np.arange(n_samples)
        np.random.shuffle(indices)

        fold_sizes = np.full(cv, n_samples // cv, dtype=int)
        fold_sizes[:n_samples % cv] += 1
        current = 0

        scores = []
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            test_indices = indices[start:stop]
            train_indices = np.concatenate([indices[:start], indices[stop:]])

            X_train, X_test = X[train_indices], X[test_indices]
            y_train, y_test = y[train_indices], y[test_indices]

            model = model_fn(X_train, y_train)

            # Ensure predict is called correctly
            y_pred = model.predict(X_test)

            score = np.mean(y_pred == y_test)
            scores.append(score)

            current = stop

        return scores

    def pca(self, n_components):
        """
        Perform Principal Component Analysis (PCA)
        """
        # Center the data
        mean = np.mean(self.data, axis=0)
        centered_data = self.data - mean

        # Compute covariance matrix
        cov_matrix = np.cov(centered_data.T)

        # Compute eigenvectors and eigenvalues
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

        # Sort eigenvectors by decreasing eigenvalues
        idx = eigenvalues.argsort()[::-1]
        eigenvectors = eigenvectors[:, idx]

        # Select top n_components eigenvectors
        components = eigenvectors[:, :n_components]

        # Project data onto principal components
        projected_data = np.dot(centered_data, components)
        out = Tensor(projected_data)

        def backward_fn(grad):
            if self.requires_grad:
                self.grad = np.dot(grad, components.T) if self.grad is None else self.grad + np.dot(grad, components.T)

        out.backward_fn = backward_fn
        return out

    def kmeans(self, n_clusters, max_iter=100):
        """
        Perform K-Means clustering with a backward pass
        """
        # Randomly initialize centroids
        centroids = self.data[np.random.choice(self.data.shape[0], n_clusters, replace=False)]

        for _ in range(max_iter):
            # Assign points to nearest centroid
            distances = np.sqrt(((self.data[:, np.newaxis, :] - centroids) ** 2).sum(axis=2))
            labels = np.argmin(distances, axis=1)

            # Update centroids
            new_centroids = np.array([self.data[labels == k].mean(axis=0) for k in range(n_clusters)])

            # Check for convergence
            if np.all(centroids == new_centroids):
                break

            centroids = new_centroids

        # Compute final distances and labels
        distances = np.sqrt(((self.data[:, np.newaxis, :] - centroids) ** 2).sum(axis=2))
        labels = np.argmin(distances, axis=1)

        out = Tensor(labels)

        def backward_fn(grad):
            if self.requires_grad:
                # Initialize gradient
                if self.grad is None:
                    self.grad = np.zeros_like(self.data, dtype=float)  # Ensure gradient is float

                # Compute gradients with respect to the input data
                for i in range(self.data.shape[0]):
                    cluster = labels[i]
                    diff = self.data[i] - centroids[cluster]
                    self.grad[i] += 2 * diff / len(labels[labels == cluster])

                # Scale the gradients by the incoming gradient
                self.grad *= grad.reshape(-1, 1)

        out.backward_fn = backward_fn
        return out, Tensor(centroids)

    def linear_regression(self, y):
        """
        Perform simple linear regression
        """
        if not isinstance(y, Tensor):
            y = Tensor(y)

        X = self.data
        y = y.data

        # Add bias term
        X_with_bias = np.column_stack((np.ones(X.shape[0]), X))

        # Compute coefficients
        coeffs = np.linalg.inv(X_with_bias.T @ X_with_bias) @ X_with_bias.T @ y

        out = Tensor(coeffs)

        def backward_fn(grad):
            if self.requires_grad:
                self.grad = grad @ X_with_bias if self.grad is None else self.grad + grad @ X_with_bias

        out.backward_fn = backward_fn
        return out

    def space_to_depth(self, block_size):
        if len(self.data.shape) != 4:
            raise ValueError("Input tensor must be 4D (batch, channels, height, width)")

        N, C, H, W = self.data.shape
        if H % block_size != 0 or W % block_size != 0:
            raise ValueError("Height and width must be divisible by block_size")

        new_C = C * (block_size ** 2)
        new_H = H // block_size
        new_W = W // block_size

        reshaped = self.data.reshape(N, C, new_H, block_size, new_W, block_size)
        transposed = reshaped.transpose(0, 1, 3, 5, 2, 4)
        output_data = transposed.reshape(N, new_C, new_H, new_W)
        out = Tensor(output_data)

        def backward_fn(grad):
            grad_reshaped = grad.reshape(N, C, block_size, block_size, new_H, new_W)
            grad_transposed = grad_reshaped.transpose(0, 1, 4, 2, 5, 3)
            grad_output = grad_transposed.reshape(N, C, H, W)
            if self.requires_grad:
                self.grad = grad_output if self.grad is None else self.grad + grad_output

        out.backward_fn = backward_fn
        return out

    def sort(self, axis=-1, descending=False):
        if descending:
            sorted_data = -np.sort(-self.data, axis=axis)
            sorted_indices = np.argsort(-self.data, axis=axis)
        else:
            sorted_data = np.sort(self.data, axis=axis)
            sorted_indices = np.argsort(self.data, axis=axis)

        out = Tensor(sorted_data)
        indices = Tensor(sorted_indices)

        def backward_fn(grad):
            unsort_indices = np.argsort(sorted_indices, axis=axis)
            if self.requires_grad:
                self.grad = np.take_along_axis(grad, unsort_indices,
                                               axis=axis) if self.grad is None else self.grad + np.take_along_axis(grad,
                                                                                                                   unsort_indices,
                                                                                                                   axis=axis)

        out.backward_fn = backward_fn
        return out, indices

    def channel_shuffle(self, groups):
        if len(self.data.shape) != 3:
            raise ValueError("Input tensor must be 3D (channels, height, width)")

        n, h, w = self.data.shape
        if n % groups != 0:
            raise ValueError("Number of channels must be divisible by the number of groups")

        reshaped = self.data.reshape((groups, n // groups, h, w))
        transposed = reshaped.transpose((1, 0, 2, 3))
        shuffled = transposed.reshape((n, h, w))
        out = Tensor(shuffled)

        def backward_fn(grad):
            grad_reshaped = grad.reshape((n // groups, groups, h, w))
            grad_transposed = grad_reshaped.transpose((1, 0, 2, 3))
            if self.requires_grad:
                self.grad = grad_transposed.reshape(
                    (n, h, w)) if self.grad is None else self.grad + grad_transposed.reshape((n, h, w))

        out.backward_fn = backward_fn
        return out

    def pixel_shuffle(self, upscale_factor):
        """
        Rearrange elements in a tensor of shape (*, C * r^2, H, W) to (*, C, H * r, W * r).

        Args:
        upscale_factor (int): Factor to increase spatial resolution by.

        Returns:
        Tensor: Rearranged tensor with increased spatial dimensions.
        """
        data = self.data

        # Ensure data is a numpy array
        if not isinstance(data, np.ndarray):
            data = np.array(data)

        # Handle different input shapes
        if len(data.shape) == 3:
            # If 3D, assume it's (C * r^2, H, W) and add a batch dimension
            data = np.expand_dims(data, axis=0)
        elif len(data.shape) != 4:
            raise ValueError("Input tensor must be 3D (C * r^2, H, W) or 4D (batch_size, C * r^2, H, W)")

        # Ensure channel dimension is second
        if data.shape[1] % (upscale_factor ** 2) != 0:
            # If channels are not in the correct position, try to rearrange
            if data.shape[-1] % (upscale_factor ** 2) == 0:
                data = np.moveaxis(data, -1, 1)
            else:
                raise ValueError("Number of channels must be divisible by upscale_factor^2")

        batch_size, channels, height, width = data.shape

        # Reshape and transpose to get pixels in target positions
        reshaped = data.reshape(batch_size, channels // (upscale_factor ** 2), upscale_factor, upscale_factor,
                                height, width)
        transposed = reshaped.transpose(0, 1, 4, 2, 5, 3)

        # Reshape to target size
        output_shape = (batch_size, channels // (upscale_factor ** 2), height * upscale_factor, width * upscale_factor)
        output = transposed.reshape(output_shape)

        out = Tensor(output)

        def backward_fn(grad):
            if self.requires_grad:
                # Reverse the operations for the backward pass
                grad_reshaped = grad.reshape(batch_size, channels // (upscale_factor ** 2), height, upscale_factor,
                                             width, upscale_factor)
                grad_transposed = grad_reshaped.transpose(0, 1, 3, 5, 2, 4)
                grad_output = grad_transposed.reshape(data.shape)

                # If original input was 3D, remove the batch dimension
                if len(self.data.shape) == 3:
                    grad_output = grad_output.squeeze(0)

                self.grad = grad_output if self.grad is None else self.grad + grad_output

        out.backward_fn = backward_fn
        return out

    def plot(self, title=None, xlabel=None, ylabel=None):
        """
        Plot the tensor data using matplotlib.

        Parameters:
        - title: str, optional title for the plot
        - xlabel: str, optional label for x-axis
        - ylabel: str, optional label for y-axis
        """
        plt.figure(figsize=(10, 6))

        if self.data.ndim == 1:
            plt.plot(self.data)
        elif self.data.ndim == 2:
            plt.imshow(self.data, cmap='viridis')
            plt.colorbar()
        else:
            raise ValueError("Can only plot 1D or 2D tensors")

        if title:
            plt.title(title)
        if xlabel:
            plt.xlabel(xlabel)
        if ylabel:
            plt.ylabel(ylabel)

        plt.show()

    def histogram(self, bins=50, title=None, xlabel=None, ylabel=None):
        """
        Plot a histogram of the tensor data.

        Parameters:
        - bins: int, number of bins for the histogram
        - title: str, optional title for the plot
        - xlabel: str, optional label for x-axis
        - ylabel: str, optional label for y-axis
        """
        plt.figure(figsize=(10, 6))
        plt.hist(self.data.flatten(), bins=bins)

        if title:
            plt.title(title)
        if xlabel:
            plt.xlabel(xlabel)
        if ylabel:
            plt.ylabel(ylabel)

        plt.show()

    def decision_tree(self, y, max_depth=None):
        """
        Custom decision tree implementation
        """

        class Node:
            def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
                self.feature = feature
                self.threshold = threshold
                self.left = left
                self.right = right
                self.value = value

        def gini_impurity(y):
            _, counts = np.unique(y, return_counts=True)
            probabilities = counts / len(y)
            return 1 - np.sum(probabilities ** 2)

        def split(X, y, feature, threshold):
            left_mask = X[:, feature] <= threshold
            right_mask = ~left_mask
            return X[left_mask], y[left_mask], X[right_mask], y[right_mask]

        def build_tree(X, y, depth=0):
            n_samples, n_features = X.shape
            n_classes = len(np.unique(y))

            if depth == max_depth or n_samples < 2 or n_classes == 1:
                return Node(value=np.argmax(np.bincount(y.flatten().astype(int))))

            best_gini = float('inf')
            best_feature = None
            best_threshold = None

            for feature in range(n_features):
                thresholds = np.unique(X[:, feature])
                for threshold in thresholds:
                    _, y_left, _, y_right = split(X, y, feature, threshold)
                    gini = (len(y_left) * gini_impurity(y_left) + len(y_right) * gini_impurity(y_right)) / n_samples
                    if gini < best_gini:
                        best_gini = gini
                        best_feature = feature
                        best_threshold = threshold

            if best_feature is None:
                return Node(value=np.argmax(np.bincount(y.astype(int))))

            X_left, y_left, X_right, y_right = split(X, y, best_feature, best_threshold)
            left = build_tree(X_left, y_left, depth + 1)
            right = build_tree(X_right, y_right, depth + 1)

            return Node(feature=best_feature, threshold=best_threshold, left=left, right=right)

        def predict_sample(x, node):
            if node.value is not None:
                return node.value
            if x[node.feature] <= node.threshold:
                return predict_sample(x, node.left)
            return predict_sample(x, node.right)

        root = build_tree(self.data, y)

        predictions = np.array([predict_sample(x, root) for x in self.data], dtype=np.int64)

        out = Tensor(predictions)

        def backward_fn(grad):
            if self.requires_grad:
                # Decision trees don't have a straightforward gradient
                # This is a placeholder for gradient computation
                self.grad = np.zeros_like(self.data) if self.grad is None else self.grad

        out.backward_fn = backward_fn
        return out, root

    def gradient_boosting(self, y, n_estimators=100, learning_rate=0.1, max_depth=3):
        """
        Custom gradient boosting implementation
        """
        y = y.reshape(-1, 1)
        trees = []

        # Initial prediction
        F = np.zeros_like(y, dtype=float)

        for _ in range(n_estimators):
            residuals = y - F
            tree, _ = self.decision_tree(residuals, max_depth=max_depth)
            trees.append(tree)
            F += learning_rate * tree.data.reshape(-1, 1)

        predictions = F.ravel()
        out = Tensor(predictions)

        def backward_fn(grad):
            if self.requires_grad:
                # Gradient boosting doesn't have a straightforward gradient
                # This is a placeholder for gradient computation
                self.grad = np.zeros_like(self.data) if self.grad is None else self.grad

        out.backward_fn = backward_fn
        return out, trees

    def confusion_matrix(self, y_true):
        """
        Compute confusion matrix
        """
        y_pred = self.data
        classes = np.unique(np.concatenate((y_true, y_pred)))
        n_classes = len(classes)
        cm = np.zeros((n_classes, n_classes), dtype=int)

        for i in range(len(y_true)):
            cm[np.where(classes == y_true[i])[0], np.where(classes == y_pred[i])[0]] += 1

        return Tensor(cm)

    def f1_score(self, y_true):
        """
        Compute F1 score
        """
        cm = self.confusion_matrix(y_true).data
        precision = np.diag(cm) / np.sum(cm, axis=0)
        recall = np.diag(cm) / np.sum(cm, axis=1)
        f1 = 2 * (precision * recall) / (precision + recall)
        return Tensor(np.nan_to_num(f1))

    def recall(self, y_true):
        """
        Compute recall
        """
        cm = self.confusion_matrix(y_true).data
        recall = np.diag(cm) / np.sum(cm, axis=1)
        return Tensor(np.nan_to_num(recall))

    def precision(self, y_true):
        """
        Compute precision
        """
        cm = self.confusion_matrix(y_true).data
        precision = np.diag(cm) / np.sum(cm, axis=0)
        return Tensor(np.nan_to_num(precision))

    def bagging(self, y, n_estimators=10, sample_size=None):
        """
        Bagging classifier implementation
        """
        if sample_size is None:
            sample_size = len(self.data)

        trees = []
        for _ in range(n_estimators):
            indices = np.random.choice(len(self.data), size=sample_size, replace=True)
            X_sample, y_sample = self.data[indices], y[indices]
            tree, _ = Tensor(X_sample).decision_tree(y_sample)
            trees.append(tree)

        def predict(X):
            predictions = np.array([tree.data for tree in trees])
            return np.apply_along_axis(lambda x: np.argmax(np.bincount(x)), axis=0, arr=predictions)

        out = Tensor(predict(self.data))

        def backward_fn(grad):
            if self.requires_grad:
                self.grad = np.zeros_like(self.data) if self.grad is None else self.grad

        out.backward_fn = backward_fn
        return out, trees

    def preprocess(self, method='standardize'):
        """
        Preprocess the tensor data using various methods.

        Parameters:
        - method: str, preprocessing method to use. Options are 'standardize', 'normalize', 'min_max_scale'

        Returns:
        - Tensor: Preprocessed tensor
        """
        if method == 'standardize':
            return self.standardize()
        elif method == 'normalize':
            return self.normalize()
        elif method == 'min_max_scale':
            return self.min_max_scale()
        else:
            raise ValueError("Invalid preprocessing method. Choose 'standardize', 'normalize', or 'min_max_scale'.")

    def cat(tensors, axis=0):
        """
        Concatenates a list of tensors along a specified axis.

        Parameters:
        - tensors: list of Tensor objects to concatenate.
        - axis: int, the axis along which to concatenate the tensors.

        Returns:
        - Tensor: A new tensor resulting from concatenating the input tensors.
        """
        # Check if all tensors require gradients
        requires_grad = any(tensor.requires_grad for tensor in tensors)

        # Extract the numpy arrays from the tensors
        data = [tensor.data for tensor in tensors]

        # Use numpy to concatenate the arrays
        concatenated_data = np.concatenate(data, axis=axis)

        # Create a new tensor from the concatenated data
        result = Tensor(concatenated_data, requires_grad=requires_grad)

        if requires_grad:
            def backward_fn(grad):
                # This function will split the gradient back to the original tensors
                grad_parts = np.split(grad, indices_or_sections=[tensor.data.shape[axis] for tensor in tensors[:-1]],
                                      axis=axis)
                for tensor, grad_part in zip(tensors, grad_parts):
                    if tensor.requires_grad:
                        if tensor.grad is None:
                            tensor.grad = grad_part
                        else:
                            tensor.grad += grad_part

            # Assign the custom backward function to handle gradients for concatenation
            result.backward_fn = backward_fn

        return result

    def full(shape, fill_value, requires_grad=False):
        """
        Returns a new tensor of given shape filled with fill_value.

        Parameters:
        - shape: int or tuple of ints, defining the shape of the new tensor.
        - fill_value: scalar, value to fill the new tensor with.
        - requires_grad: bool, whether the new tensor requires gradient.

        Returns:
        - Tensor: A new tensor with the specified shape and fill value.
        """
        # Use numpy to create an array filled with the fill value
        data = np.full(shape, fill_value)

        # Create and return a new Tensor object
        return Tensor(data, requires_grad=requires_grad)

    def __repr__(self):
        return f"Tensor({self.data})"

    def backward(self):
        if self.backward_fn is None:
            raise RuntimeError("No backward function defined for this tensor.")
        self.backward_fn(np.ones_like(self.data))


class Dataset:
    def __init__(self, data: Union[np.ndarray, List[np.ndarray]]):
        if isinstance(data, np.ndarray):
            self.data = [data]
        elif isinstance(data, list) and all(isinstance(item, np.ndarray) for item in data):
            self.data = data
        else:
            raise ValueError("Data must be a numpy array or a list of numpy arrays")

        self.tensors = [Tensor(arr) for arr in self.data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.tensors[idx]

    def to_tensors(self) -> List[Tensor]:
        return self.tensors

    def batch(self, batch_size: int) -> List[List[Tensor]]:
        batches = []
        for i in range(0, len(self), batch_size):
            batches.append(self.tensors[i:i + batch_size])
        return batches

    def shuffle(self):
        indices = np.random.permutation(len(self))
        self.data = [self.data[i] for i in indices]
        self.tensors = [self.tensors[i] for i in indices]

    def split(self, split_ratio: float) -> Tuple['Dataset', 'Dataset']:
        split_idx = int(len(self) * split_ratio)
        return Dataset(self.data[:split_idx]), Dataset(self.data[split_idx:])

    @classmethod
    def from_tensor_list(cls, tensor_list: List[Tensor]):
        return cls([tensor.data for tensor in tensor_list])

    @classmethod
    def from_numpy(cls, *arrays):
        return cls(list(arrays))

    def apply(self, func):
        self.data = [func(arr) for arr in self.data]
        self.tensors = [Tensor(arr) for arr in self.data]

    def __repr__(self):
        return f"Dataset(num_tensors={len(self)}, shapes={[arr.shape for arr in self.data]})"


class DenseLayer:
    def __init__(self, input_size, output_size):
        """
        Paramaters:
        self input_size = the size of an input
        self output_size
        self weights = randomized input size and output_size with the numpy random algorithm * 0,1
        """
        self.weights = Tensor(np.random.randn(input_size, output_size) * 0.1)
        self.biases = Tensor(np.zeros((1, output_size)))

    def forward(self, inputs: Tensor):
        self.inputs = inputs
        self.outputs = inputs.dot(self.weights) + self.biases
        return self.outputs

    def backward(self, dL_dout: Tensor, lr: float):
        dL_dweights = self.inputs.transpose().dot(dL_dout)
        dL_dbiases = np.sum(dL_dout.data, axis=0, keepdims=True)
        dL_dinputs = dL_dout.dot(Tensor(self.weights.data.T))

        self.weights.grad = dL_dweights.data if self.weights.grad is None else self.weights.grad + dL_dweights.data
        self.biases.grad = dL_dbiases if self.biases.grad is None else self.biases.grad + dL_dbiases
        self.inputs.backward_fn = lambda grad: grad + dL_dinputs.data if self.inputs.backward_fn is None else lambda \
                x: self.inputs.backward_fn(x) + dL_dinputs.data

        self.weights.data -= lr * self.weights.grad
        self.biases.data -= lr * self.biases.grad

        return dL_dinputs

    def get_params(self):
        return {'weights': self.weights.data, 'biases': self.biases.data}

    def set_params(self, params):
        self.weights.data = params['weights']
        self.biases.data = params['biases']


class ELUActivationLayer:
    """
    Paramaters:
    self alpha = the alpha value
    self inputs = the inputs
    self outputs = the outputs
    """

    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def forward(self, inputs: Tensor):
        self.inputs = inputs
        self.outputs = inputs.apply(lambda x: x if x > 0 else self.alpha * (np.exp(
            x) - 1))  # If x is greater than 0, return x. If x is less than 0, return alpha * (exp(x) - 1)
        return self.outputs

    def backward(self, dL_dout: Tensor, lr: float):
        def elu_derivative(x):
            if x > 0:
                return 1
            else:
                return self.alpha * np.exp(x)

        dL_dinputs = dL_dout * self.inputs.apply(elu_derivative)
        self.inputs.grad = dL_dinputs.data if self.inputs.grad is None else self.inputs.grad + dL_dinputs.data  # Add the dL_dinputs to the grad
        self.inputs.backward_fn = lambda grad: grad + dL_dinputs.data if self.inputs.backward_fn is None else lambda \
                x: self.inputs.backward_fn(x) + dL_dinputs.data  # Add the dL_dinputs to the backward function
        return dL_dinputs

    def get_params(self):
        return None

    def set_params(self, params):
        pass


class ReLUActivationLayer:
    """
    Paramaters:
    self inputs = the inputs
    self outputs = the outputs
    """

    def forward(self, inputs: Tensor):
        self.inputs = inputs
        self.outputs = inputs.apply(lambda x: max(0, x))
        return self.outputs

    def backward(self, dL_dout: Tensor, lr: float):
        dL_dinputs = dL_dout * self.inputs.apply(
            lambda x: 1 if x > 0 else 0)  # If x is greater than 0, return 1. If x is less than 0, return 0

        self.inputs.grad = dL_dinputs.data if self.inputs.grad is None else self.inputs.grad + dL_dinputs.data
        self.inputs.backward_fn = lambda grad: grad + dL_dinputs.data if self.inputs.backward_fn is None else lambda \
                x: self.inputs.backward_fn(x) + dL_dinputs.data  # Add the dL_dinputs to the backward function

        return dL_dinputs

    def get_params(self):
        return None

    def set_params(self, params):
        pass


class SELUActivationLayer:
    """
    Parameters:
    self.inputs = the inputs
    self.outputs = the outputs
    """

    def __init__(self, alpha=1.6732632423543772848170429916717, scale=1.0507009873554804934193349852946):
        self.alpha = alpha
        self.scale = scale

    def forward(self, inputs: Tensor):
        self.inputs = inputs
        self.outputs = inputs.apply(lambda x: self.scale * (x if x > 0 else self.alpha * (np.exp(x) - 1)))
        return self.outputs

    def backward(self, dL_dout: Tensor, lr: float):
        dL_dinputs = dL_dout * self.inputs.apply(
            lambda x: self.scale if x > 0 else self.scale * self.alpha * np.exp(x))

        self.inputs.grad = dL_dinputs.data if self.inputs.grad is None else self.inputs.grad + dL_dinputs.data
        self.inputs.backward_fn = lambda grad: grad + dL_dinputs.data if self.inputs.backward_fn is None else lambda \
                x: self.inputs.backward_fn(x) + dL_dinputs.data

        return dL_dinputs

    def get_params(self):
        return None

    def set_params(self, params):
        pass


class LeakyReLUActivationLayer:
    """
    Paramaters:
    self inputs = the inputs
    self outputs = the outputs
    self alpha = the alpha value

    It is simmilar to ReLUActivationLayer
    """

    def __init__(self, alpha: float = 0.01):
        self.alpha = alpha

    def forward(self, inputs: Tensor):
        self.inputs = inputs
        self.outputs = inputs.apply(lambda x: x if x > 0 else self.alpha * x)
        return self.outputs

    def backward(self, dL_dout: Tensor, lr: float):
        dL_dinputs = dL_dout * self.inputs.apply(lambda x: 1 if x > 0 else self.alpha)

        self.inputs.grad = dL_dinputs.data if self.inputs.grad is None else self.inputs.grad + dL_dinputs.data
        self.inputs.backward_fn = lambda grad: grad + dL_dinputs.data if self.inputs.backward_fn is None else lambda \
                x: self.inputs.backward_fn(x) + dL_dinputs.data

        return dL_dinputs

    def get_params(self):
        return None

    def set_params(self, params):
        pass


class PReLUActivationLayer:
    """
    Paramaters:
    self inputs = the inputs
    self outputs = the outputs
    self alpha = the alpha value
    """

    def __init__(self, alpha: float = 0.01):
        self.alpha = alpha
        self.d_alpha = 0

    def forward(self, inputs: Tensor):
        self.inputs = inputs
        self.outputs = inputs.apply(lambda x: x if x > 0 else self.alpha * x)
        return self.outputs

    def backward(self, dL_dout: Tensor, lr: float):
        dL_dinputs = dL_dout * self.inputs.apply(lambda x: 1 if x > 0 else self.alpha)

        grad_alpha = dL_dout * self.inputs.apply(lambda x: x if x <= 0 else 0)
        self.d_alpha = np.sum(grad_alpha.data)
        self.alpha -= lr * self.d_alpha

        if self.inputs.grad is None:
            self.inputs.grad = dL_dinputs.data
        else:
            self.inputs.grad += dL_dinputs.data

        if self.inputs.backward_fn is None:
            self.inputs.backward_fn = lambda grad: dL_dinputs.data  # Add the dL_dinputs to the backward function
        else:
            prev_backward_fn = self.inputs.backward_fn
            self.inputs.backward_fn = lambda grad: prev_backward_fn(
                grad) + dL_dinputs.data  # Add the dL_dinputs to the backward function

        return dL_dinputs

    def get_params(self):
        return {'alpha': self.alpha}

    def set_params(self, params):
        if params and 'alpha' in params:
            self.alpha = params['alpha']


class SoftsignActivationLayer:
    def forward(self, inputs: Tensor):
        self.inputs = inputs
        self.outputs = Tensor(inputs.data / (1 + np.abs(inputs.data)))
        return self.outputs

    def backward(self, dL_dout: Tensor, lr: float = None):
        gradients = dL_dout.data * self._compute_derivative()
        return Tensor(gradients)

    def _compute_derivative(self):
        return 1 / ((1 + np.abs(self.inputs.data)) ** 2)

    def get_params(self):
        return None

    def set_params(self, params: dict):
        pass


class SoftmaxActivationLayer:
    """
    Paramaters:
    self inputs = the inputs
    self outputs = the outputs
    """

    def forward(self, inputs: Tensor):
        exp_values = np.exp(inputs.data - np.max(inputs.data, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1,
                                            keepdims=True)  # probabilities is equal to exp_values / sum(exp_values, axis=1, keepdims=True)
        self.outputs = Tensor(probabilities)
        return self.outputs

    def backward(self, dL_dout: Tensor, lr: float = None):
        jacobians = self._compute_jacobians()

        gradients = np.einsum('ijk,ik->ij', jacobians, dL_dout.data)  # gradients is equal to jacobians * dL_dout

        return Tensor(gradients)

    def _compute_jacobians(self):
        batch_size, num_classes = self.outputs.data.shape
        jacobians = np.zeros((batch_size, num_classes, num_classes))

        for i in range(batch_size):
            for j in range(num_classes):
                for k in range(num_classes):
                    if j == k:
                        jacobians[i, j, k] = self.outputs.data[i, j] * (
                                1 - self.outputs.data[i, k])  # jacobian is equal to outputs * (1 - outputs)
                    else:
                        jacobians[i, j, k] = -self.outputs.data[i, j] * self.outputs.data[
                            i, k]  # jacobian is equal to -outputs * outputs

        return jacobians

    def get_params(self):
        return None

    def set_params(self, params: dict):
        pass


class Fold:
    def __init__(self, output_size, kernel_size, stride=1, padding=0):
        self.output_size = output_size
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.input_shape = None

    def forward(self, inputs: Tensor):
        self.input_shape = inputs.data.shape
        batch_size, n_channels, length = inputs.data.shape

        # Calculate output dimensions
        height, width = self.output_size
        out_height = (height + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_width = (width + 2 * self.padding - self.kernel_size) // self.stride + 1

        # Reshape the input
        folded = inputs.data.reshape(batch_size, n_channels // (self.kernel_size ** 2),
                                     self.kernel_size, self.kernel_size, out_height, out_width)

        # Transpose and reshape to get the final output
        folded = folded.transpose(0, 1, 4, 2, 5, 3).reshape(batch_size, -1, height, width)

        return Tensor(folded)

    def backward(self, dL_dout: Tensor):
        batch_size, _, height, width = dL_dout.data.shape

        # Calculate output dimensions
        out_height = (height + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_width = (width + 2 * self.padding - self.kernel_size) // self.stride + 1

        # Reshape to match the unfolded shape
        grad_reshaped = dL_dout.data.reshape(batch_size, -1, out_height, self.kernel_size, out_width, self.kernel_size)
        grad_transposed = grad_reshaped.transpose(0, 1, 3, 5, 2, 4)

        # Reshape to match the input shape
        unfolded = grad_transposed.reshape(self.input_shape)

        return Tensor(unfolded)


class Unfold:
    def __init__(self, kernel_size, stride=1, padding=0):
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.input_shape = None

    def forward(self, inputs: Tensor):
        self.input_shape = inputs.data.shape
        batch_size, n_channels, height, width = inputs.data.shape

        # Calculate output dimensions
        out_height = (height + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_width = (width + 2 * self.padding - self.kernel_size) // self.stride + 1

        # Perform the unfolding operation
        unfolded = np.zeros((batch_size, n_channels * self.kernel_size * self.kernel_size,
                             out_height * out_width))

        for i in range(out_height):
            for j in range(out_width):
                h_start = i * self.stride - self.padding
                w_start = j * self.stride - self.padding
                h_end = h_start + self.kernel_size
                w_end = w_start + self.kernel_size

                # Extract the patch
                patch = inputs.data[:, :, max(0, h_start):min(height, h_end),
                        max(0, w_start):min(width, w_end)]

                # Pad the patch if necessary
                if patch.shape[2] < self.kernel_size or patch.shape[3] < self.kernel_size:
                    padded_patch = np.zeros((batch_size, n_channels, self.kernel_size, self.kernel_size))
                    padded_patch[:, :, :patch.shape[2], :patch.shape[3]] = patch
                    patch = padded_patch

                # Flatten the patch and store it in the output
                unfolded[:, :, i * out_width + j] = patch.reshape(batch_size, -1)

        return Tensor(unfolded)

    def backward(self, dL_dout: Tensor):
        batch_size, _, _ = dL_dout.data.shape
        _, n_channels, height, width = self.input_shape

        # Initialize the gradient with respect to input
        dx = np.zeros(self.input_shape)

        # Calculate output dimensions
        out_height = (height + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_width = (width + 2 * self.padding - self.kernel_size) // self.stride + 1

        # Reshape dL_dout
        dL_dout = dL_dout.data.reshape(batch_size, n_channels, self.kernel_size, self.kernel_size,
                                       out_height, out_width)

        # Perform the folding operation
        for i in range(out_height):
            for j in range(out_width):
                h_start = i * self.stride - self.padding
                w_start = j * self.stride - self.padding
                h_end = h_start + self.kernel_size
                w_end = w_start + self.kernel_size

                # Add the gradient to the appropriate location
                dx[:, :, max(0, h_start):min(height, h_end),
                max(0, w_start):min(width, w_end)] += dL_dout[:, :, :, :, i, j]

        return Tensor(dx)


class FlattenLayer:
    def __init__(self):
        self.inputs_shape = None

    def forward(self, inputs: Tensor):
        self.inputs_shape = inputs.data.shape
        batch_size = inputs.data.shape[0]
        flattened_shape = (batch_size,
                           -1)  # Reshape shape by -1 leaving the batch_size dimension intact and flattening the remaining dimensions
        return inputs.reshape(flattened_shape)

    def backward(self, dL_dout: Tensor, lr: float):
        return dL_dout.reshape(self.inputs_shape)


class UnflattenLayer:
    def __init__(self, output_shape):
        self.output_shape = output_shape

    def forward(self, inputs: Tensor):
        batch_size = inputs.data.shape[0]
        unflattened_shape = (batch_size,) + self.output_shape
        return inputs.reshape(unflattened_shape)

    def backward(self, dL_dout: Tensor, lr: float):
        batch_size = dL_dout.data.shape[0]
        flattened_shape = (batch_size, -1)  # Doing the opposite of flattening
        return dL_dout.reshape(flattened_shape)


class Conv1DLayer:
    """
    Parameters:
    self weights = the weights
    self biases = the biases
    self in channels = input channels
    self out channels = output channels
    self padding = the padding
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.weights = Tensor(np.random.randn(out_channels, in_channels, kernel_size).astype(
            np.float32) * 0.1)  # weights is equal to np.random.randn(out_channels, in_channels, kernel_size).astype(np.float32) * 0.1
        self.biases = Tensor(np.zeros((out_channels, 1),
                                      dtype=np.float32))  # biases is equal to np.zeros((out_channels, 1), dtype=np.float32)

    def forward(self, inputs: Tensor):
        self.inputs = inputs
        batch_size, in_channels, input_length = inputs.data.shape

        output_length = ((input_length + 2 * self.padding - self.kernel_size) // self.stride) + 1

        if self.padding > 0:
            padded_inputs = np.pad(inputs.data, ((0, 0), (0, 0), (self.padding, self.padding)), mode='constant')
        else:
            padded_inputs = inputs.data

        output = np.zeros((batch_size, self.out_channels, output_length), dtype=np.float32)
        for i in range(output_length):
            start = i * self.stride
            end = start + self.kernel_size
            output[:, :, i] = np.sum(
                padded_inputs[:, np.newaxis, :, start:end] * self.weights.data[np.newaxis, :, :, :],
                axis=(2, 3)
            )
            # Sums the product of the padded inputs and the weights along an axis

        output += self.biases.data.reshape(1, -1, 1)

        self.output = Tensor(output)
        return self.output

    def backward(self, dL_dout: Tensor, lr: float):
        batch_size, _, output_length = dL_dout.data.shape

        dL_dweights = np.zeros_like(self.weights.data)
        dL_dbiases = np.sum(dL_dout.data, axis=(0, 2), keepdims=True)
        dL_dinputs = np.zeros_like(self.inputs.data)

        if self.padding > 0:
            padded_inputs = np.pad(self.inputs.data, ((0, 0), (0, 0), (self.padding, self.padding)), mode='constant')
        else:
            padded_inputs = self.inputs.data

        for i in range(output_length):
            start = i * self.stride
            end = start + self.kernel_size
            dL_dweights += np.sum(
                padded_inputs[:, np.newaxis, :, start:end] * dL_dout.data[:, :, i:i + 1, np.newaxis],
                axis=0
            )
            dL_dinputs[:, :, start:end] += np.sum(
                self.weights.data[np.newaxis, :, :, :] * dL_dout.data[:, :, i:i + 1, np.newaxis],
                axis=1
            )

        if self.padding > 0:
            dL_dinputs = dL_dinputs[:, :, self.padding:-self.padding]

        self.weights.grad = dL_dweights if self.weights.grad is None else self.weights.grad + dL_dweights
        self.biases.grad = dL_dbiases if self.biases.grad is None else self.biases.grad + dL_dbiases

        self.weights.data -= lr * self.weights.grad
        self.biases.data -= lr * self.biases.grad

        self.inputs.grad = dL_dinputs if self.inputs.grad is None else self.inputs.grad + dL_dinputs
        self.inputs.backward_fn = lambda grad: grad + dL_dinputs if self.inputs.backward_fn is None else lambda \
                x: self.inputs.backward_fn(x) + dL_dinputs

        return Tensor(dL_dinputs)

    def get_params(self):
        return {
            'weights': self.weights.data,
            'biases': self.biases.data
        }

    def set_params(self, params):
        self.weights.data = params['weights']
        self.biases.data = params['biases']


class Conv2DLayer:
    # The same as Conv1d but multiplied along 2 dimensions
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, transposed=False):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.transposed = transposed

        if transposed:
            self.filters = Tensor(np.random.randn(in_channels, out_channels, *self.kernel_size) * 0.1)
        else:
            self.filters = Tensor(np.random.randn(out_channels, in_channels, *self.kernel_size) * 0.1)
        self.biases = Tensor(np.zeros((out_channels, 1)))

    def forward(self, inputs: Tensor):
        self.inputs = inputs
        if self.transposed:
            return self.forward_transposed(inputs)
        else:
            return self.forward_normal(inputs)

    def forward_normal(self, inputs: Tensor):
        batch_size, in_channels, in_height, in_width = inputs.data.shape
        out_height = (in_height + 2 * self.padding[0] - self.kernel_size[0]) // self.stride[0] + 1
        out_width = (in_width + 2 * self.padding[1] - self.kernel_size[1]) // self.stride[1] + 1

        padded_inputs = np.pad(inputs.data,
                               ((0, 0), (0, 0), (self.padding[0], self.padding[0]), (self.padding[1], self.padding[1])),
                               mode='constant')

        output = np.zeros((batch_size, self.out_channels, out_height, out_width))

        for i in range(out_height):
            for j in range(out_width):
                h_start = i * self.stride[0]
                h_end = h_start + self.kernel_size[0]
                w_start = j * self.stride[1]
                w_end = w_start + self.kernel_size[1]

                receptive_field = padded_inputs[:, :, h_start:h_end, w_start:w_end]
                output[:, :, i, j] = np.sum(receptive_field[:, np.newaxis, :, :, :] * self.filters.data, axis=(2, 3, 4))

        output += self.biases.data.reshape(1, -1, 1, 1)
        self.outputs = Tensor(output)
        return self.outputs

    def forward_transposed(self, inputs: Tensor):
        batch_size, in_channels, in_height, in_width = inputs.data.shape
        out_height = (in_height - 1) * self.stride[0] - 2 * self.padding[0] + self.kernel_size[0]
        out_width = (in_width - 1) * self.stride[1] - 2 * self.padding[1] + self.kernel_size[1]

        output = np.zeros((batch_size, self.out_channels, out_height, out_width))

        for i in range(in_height):
            for j in range(in_width):
                h_start = i * self.stride[0]
                h_end = h_start + self.kernel_size[0]
                w_start = j * self.stride[1]
                w_end = w_start + self.kernel_size[1]

                output[:, :, h_start:h_end, w_start:w_end] += np.sum(
                    inputs.data[:, :, i, j][:, :, np.newaxis, np.newaxis, np.newaxis] * self.filters.data,
                    axis=1
                )

        if self.padding[0] > 0 or self.padding[1] > 0:
            output = output[:, :, self.padding[0]:output.shape[2] - self.padding[0],
                     self.padding[1]:output.shape[3] - self.padding[1]]

        output += self.biases.data.reshape(1, -1, 1, 1)
        self.outputs = Tensor(output)
        return self.outputs

    def backward(self, dL_dout: Tensor, lr: float):
        if self.transposed:
            return self.backward_transposed(dL_dout, lr)
        else:
            return self.backward_normal(dL_dout, lr)

    def backward_normal(self, dL_dout: Tensor, lr: float):
        batch_size, _, out_height, out_width = dL_dout.data.shape
        padded_inputs = np.pad(self.inputs.data,
                               ((0, 0), (0, 0), (self.padding[0], self.padding[0]), (self.padding[1], self.padding[1])),
                               mode='constant')

        dL_dfilters = np.zeros_like(self.filters.data)
        dL_dinputs = np.zeros_like(padded_inputs)

        for i in range(out_height):
            for j in range(out_width):
                h_start = i * self.stride[0]
                h_end = h_start + self.kernel_size[0]
                w_start = j * self.stride[1]
                w_end = w_start + self.kernel_size[1]

                receptive_field = padded_inputs[:, :, h_start:h_end, w_start:w_end]
                dL_dfilters += np.sum(receptive_field[:, np.newaxis, :, :, :] * dL_dout.data[:, :, i:i + 1, j:j + 1],
                                      axis=0)
                dL_dinputs[:, :, h_start:h_end, w_start:w_end] += np.sum(
                    self.filters.data[np.newaxis, :, :, :, :] * dL_dout.data[:, :, i:i + 1, j:j + 1], axis=1)

        dL_dbiases = np.sum(dL_dout.data, axis=(0, 2, 3), keepdims=True)

        self.filters.grad = dL_dfilters if self.filters.grad is None else self.filters.grad + dL_dfilters
        self.biases.grad = dL_dbiases if self.biases.grad is None else self.biases.grad + dL_dbiases
        self.inputs.backward_fn = lambda grad: grad + dL_dinputs if self.inputs.backward_fn is None else lambda \
                x: self.inputs.backward_fn(x) + dL_dinputs

        self.filters.data -= lr * self.filters.grad
        self.biases.data -= lr * self.biases.grad

        if self.padding[0] > 0 or self.padding[1] > 0:
            dL_dinputs = dL_dinputs[:, :, self.padding[0]:-self.padding[0], self.padding[1]:-self.padding[1]]

        return Tensor(dL_dinputs)

    def backward_transposed(self, dL_dout: Tensor, lr: float):
        batch_size, _, out_height, out_width = dL_dout.data.shape
        in_height = (out_height + 2 * self.padding[0] - self.kernel_size[0]) // self.stride[0] + 1
        in_width = (out_width + 2 * self.padding[1] - self.kernel_size[1]) // self.stride[1] + 1

        dL_dfilters = np.zeros_like(self.filters.data)
        dL_dinputs = np.zeros((batch_size, self.in_channels, in_height, in_width))

        padded_dL_dout = np.pad(dL_dout.data, (
            (0, 0), (0, 0), (self.padding[0], self.padding[0]), (self.padding[1], self.padding[1])), mode='constant')

        for i in range(in_height):
            for j in range(in_width):
                h_start = i * self.stride[0]
                h_end = h_start + self.kernel_size[0]
                w_start = j * self.stride[1]
                w_end = w_start + self.kernel_size[1]

                dL_dfilters += np.sum(
                    self.inputs.data[:, :, i:i + 1, j:j + 1][:, :, np.newaxis, np.newaxis, np.newaxis] * padded_dL_dout[
                                                                                                         :, np.newaxis,
                                                                                                         :,
                                                                                                         h_start:h_end,
                                                                                                         w_start:w_end],
                    axis=0
                )
                dL_dinputs[:, :, i, j] = np.sum(
                    self.filters.data * padded_dL_dout[:, np.newaxis, :, h_start:h_end, w_start:w_end],
                    axis=(2, 3, 4)
                )

        dL_dbiases = np.sum(dL_dout.data, axis=(0, 2, 3), keepdims=True)

        self.filters.grad = dL_dfilters if self.filters.grad is None else self.filters.grad + dL_dfilters
        self.biases.grad = dL_dbiases if self.biases.grad is None else self.biases.grad + dL_dbiases
        self.inputs.backward_fn = lambda grad: grad + dL_dinputs if self.inputs.backward_fn is None else lambda \
                x: self.inputs.backward_fn(x) + dL_dinputs

        self.filters.data -= lr * self.filters.grad
        self.biases.data -= lr * self.biases.grad

        return Tensor(dL_dinputs)

    def get_params(self):
        return {'filters': self.filters.data, 'biases': self.biases.data}

    def set_params(self, params):
        self.filters.data = params['filters']
        self.biases.data = params['biases']


class Conv3DLayer:
    # The same as conv1d but with *3 dimensions
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding, padding)

        self.filters = Tensor(np.random.randn(out_channels, in_channels, *self.kernel_size).astype(np.float32) * 0.1)
        self.biases = Tensor(np.zeros((out_channels, 1), dtype=np.float32))

    def forward(self, inputs: Tensor):
        self.inputs = inputs
        batch_size, in_channels, in_depth, in_height, in_width = inputs.data.shape

        out_depth = ((in_depth + 2 * self.padding[0] - self.kernel_size[0]) // self.stride[0]) + 1
        out_height = ((in_height + 2 * self.padding[1] - self.kernel_size[1]) // self.stride[1]) + 1
        out_width = ((in_width + 2 * self.padding[2] - self.kernel_size[2]) // self.stride[2]) + 1

        if any(p > 0 for p in self.padding):
            padded_inputs = np.pad(inputs.data, ((0, 0), (0, 0),
                                                 (self.padding[0], self.padding[0]),
                                                 (self.padding[1], self.padding[1]),
                                                 (self.padding[2], self.padding[2])), mode='constant')
        else:
            padded_inputs = inputs.data

        output = np.zeros((batch_size, self.out_channels, out_depth, out_height, out_width), dtype=np.float32)

        for i in range(out_depth):
            for j in range(out_height):
                for k in range(out_width):
                    d_start = i * self.stride[0]
                    d_end = d_start + self.kernel_size[0]
                    h_start = j * self.stride[1]
                    h_end = h_start + self.kernel_size[1]
                    w_start = k * self.stride[2]
                    w_end = w_start + self.kernel_size[2]

                    output[:, :, i, j, k] = np.sum(
                        padded_inputs[:, np.newaxis, :, d_start:d_end, h_start:h_end, w_start:w_end] *
                        self.filters.data[np.newaxis, :, :, :, :, :],
                        axis=(2, 3, 4, 5)
                    )

        output += self.biases.data.reshape(1, -1, 1, 1, 1)

        self.output = Tensor(output)
        return self.output

    def backward(self, dL_dout: Tensor, lr: float):
        batch_size, _, out_depth, out_height, out_width = dL_dout.data.shape

        dL_dfilters = np.zeros_like(self.filters.data)
        dL_dbiases = np.sum(dL_dout.data, axis=(0, 2, 3, 4), keepdims=True)
        dL_dinputs = np.zeros_like(self.inputs.data)

        if any(p > 0 for p in self.padding):
            padded_inputs = np.pad(self.inputs.data, ((0, 0), (0, 0),
                                                      (self.padding[0], self.padding[0]),
                                                      (self.padding[1], self.padding[1]),
                                                      (self.padding[2], self.padding[2])), mode='constant')
        else:
            padded_inputs = self.inputs.data

        for i in range(out_depth):
            for j in range(out_height):
                for k in range(out_width):
                    d_start = i * self.stride[0]
                    d_end = d_start + self.kernel_size[0]
                    h_start = j * self.stride[1]
                    h_end = h_start + self.kernel_size[1]
                    w_start = k * self.stride[2]
                    w_end = w_start + self.kernel_size[2]

                    dL_dfilters += np.sum(
                        padded_inputs[:, np.newaxis, :, d_start:d_end, h_start:h_end, w_start:w_end] *
                        dL_dout.data[:, :, i:i + 1, j:j + 1, k:k + 1],
                        axis=0
                    )
                    dL_dinputs[:, :, d_start:d_end, h_start:h_end, w_start:w_end] += np.sum(
                        self.filters.data[np.newaxis, :, :, :, :, :] *
                        dL_dout.data[:, :, i:i + 1, j:j + 1, k:k + 1],
                        axis=1
                    )

        if any(p > 0 for p in self.padding):
            dL_dinputs = dL_dinputs[:, :,
                         self.padding[0]:-self.padding[0],
                         self.padding[1]:-self.padding[1],
                         self.padding[2]:-self.padding[2]]

        self.filters.grad = dL_dfilters if self.filters.grad is None else self.filters.grad + dL_dfilters
        self.biases.grad = dL_dbiases if self.biases.grad is None else self.biases.grad + dL_dbiases

        self.filters.data -= lr * self.filters.grad
        self.biases.data -= lr * self.biases.grad

        self.inputs.grad = dL_dinputs if self.inputs.grad is None else self.inputs.grad + dL_dinputs
        self.inputs.backward_fn = lambda grad: grad + dL_dinputs if self.inputs.backward_fn is None else lambda \
                x: self.inputs.backward_fn(x) + dL_dinputs

        return Tensor(dL_dinputs)

    def get_params(self):
        return {
            'filters': self.filters.data,
            'biases': self.biases.data
        }

    def set_params(self, params):
        self.filters.data = params['filters']
        self.biases.data = params['biases']


class L1RegularizationLayer:
    """
    Parameters:
    -----------
    self layer: Layer
    self lambda_: lambda_
    """

    def __init__(self, layer, lambda_):
        self.layer = layer
        self.lambda_ = lambda_

    def forward(self, inputs: Tensor):
        return self.layer.forward(inputs)

    def backward(self, dL_dout: Tensor, lr: float):
        dL_dx = self.layer.backward(dL_dout, lr)

        l1_grad_gamma = self.lambda_ * np.sign(self.layer.gamma.data)
        l1_grad_beta = self.lambda_ * np.sign(self.layer.beta.data)

        self.layer.gamma.grad += l1_grad_gamma.reshape(
            self.layer.gamma.grad.shape)  # add gradient to +=  reshaped l1_grad
        self.layer.beta.grad += l1_grad_beta.reshape(self.layer.beta.grad.shape)

        self.layer.gamma.data -= lr * l1_grad_gamma.reshape(self.layer.gamma.data.shape)
        self.layer.beta.data -= lr * l1_grad_beta.reshape(self.layer.beta.data.shape)

        return dL_dx

    def get_params(self):
        return self.layer.get_params()

    def set_params(self, params):
        self.layer.set_params(params)


class L2RegularizationLayer:
    # The same as in L1 but with 2 dimensions
    def __init__(self, layer, lambda_):
        self.layer = layer
        self.lambda_ = lambda_

    def forward(self, inputs):
        return self.layer.forward(inputs)

    def backward(self, dL_dout, lr):
        dL_dx = self.layer.backward(dL_dout, lr)

        l2_grad_gamma = 2 * self.lambda_ * self.layer.gamma.data
        l2_grad_beta = 2 * self.lambda_ * self.layer.beta.data

        self.layer.gamma.grad += l2_grad_gamma.reshape(self.layer.gamma.grad.shape)
        self.layer.beta.grad += l2_grad_beta.reshape(self.layer.beta.grad.shape)

        self.layer.gamma.data -= lr * l2_grad_gamma.reshape(self.layer.gamma.data.shape)
        self.layer.beta.data -= lr * l2_grad_beta.reshape(self.layer.beta.data.shape)

        return dL_dx

    def get_params(self):
        return self.layer.get_params()

    def set_params(self, params):
        self.layer.set_params(params)


class L3RegularizationLayer:
    # The same as in L1 but with 3 dimensions
    def __init__(self, layer, lambda_):
        self.layer = layer
        self.lambda_ = lambda_

    def forward(self, inputs):
        return self.layer.forward(inputs)

    def backward(self, dL_dout, lr):
        dL_dx = self.layer.backward(dL_dout, lr)

        l3_grad_gamma = 3 * self.lambda_ * self.layer.gamma.data ** 2
        l3_grad_beta = 3 * self.lambda_ * self.layer.beta.data ** 2

        self.layer.gamma.grad += l3_grad_gamma.reshape(self.layer.gamma.grad.shape)
        self.layer.beta.grad += l3_grad_beta.reshape(self.layer.beta.grad.shape)

        self.layer.gamma.data -= lr * l3_grad_gamma.reshape(self.layer.gamma.data.shape)
        self.layer.beta.data -= lr * l3_grad_beta.reshape(self.layer.beta.data.shape)

        return dL_dx

    def get_params(self):
        return self.layer.get_params()

    def set_params(self, params):
        self.layer.set_params(params)


class BatchNormLayer:
    """
    Parameters:
    -----------
    self num_features: num_features
    self eps: eps
    self momentum: momentum
    self gamma: gamma
    self.beta: beta
    self.running_mean: running_mean
    self.running_var: running_var
    self.training: training

    Explanations:
    ------------
    self.training: If True, the layer is in training mode, else in evaluation mode
    self.gamma and self.beta are learnable parameters
    self.momentum is a hyperparameter
    self.eps is a small value to avoid division by zero
    self.running_mean and self.running_var are running estimates of the mean and variance
    self.x_centered is the centered input
    self.x_norm is the normalized input
    self.outputs is the output of the layer
    self.num_features is the number of features in the input
    """

    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        self.gamma = Tensor(np.ones((1, num_features)))
        self.beta = Tensor(np.zeros((1, num_features)))

        self.running_mean = np.zeros((1, num_features))
        self.running_var = np.ones((1, num_features))

        self.training = True

    def forward(self, inputs: Tensor):
        self.inputs = inputs
        self.batch_size, _ = inputs.data.shape

        if self.training:
            mean = np.mean(inputs.data, axis=0)
            var = np.var(inputs.data, axis=0)

            self.running_mean = self.momentum * mean + (1 - self.momentum) * self.running_mean
            self.running_var = self.momentum * var + (1 - self.momentum) * self.running_var
        else:
            mean = self.running_mean
            var = self.running_var

        self.x_centered = inputs.data - mean
        self.x_norm = self.x_centered / np.sqrt(var + self.eps)

        outputs = self.gamma.data * self.x_norm + self.beta.data

        self.outputs = Tensor(outputs)
        return self.outputs

    def backward(self, dL_dout: Tensor, lr: float):
        dL_dgamma = np.sum(dL_dout.data * self.x_norm, axis=0)
        dL_dbeta = np.sum(dL_dout.data, axis=0)

        dL_dx_norm = dL_dout.data * self.gamma.data

        dL_dvar = np.sum(dL_dx_norm * self.x_centered * -0.5 * (self.running_var + self.eps) ** (-3 / 2), axis=0)

        dL_dmean = np.sum(dL_dx_norm * -1 / np.sqrt(self.running_var + self.eps), axis=0)
        dL_dmean += dL_dvar * np.mean(-2 * self.x_centered, axis=0)

        dL_dx = dL_dx_norm / np.sqrt(self.running_var + self.eps)
        dL_dx += dL_dvar * 2 * self.x_centered / self.batch_size
        dL_dx += dL_dmean / self.batch_size

        self.gamma.grad = dL_dgamma if self.gamma.grad is None else self.gamma.grad + dL_dgamma
        self.beta.grad = dL_dbeta if self.beta.grad is None else self.beta.grad + dL_dbeta
        self.inputs.backward_fn = lambda grad: grad + dL_dx if self.inputs.backward_fn is None else lambda \
                x: self.inputs.backward_fn(x) + dL_dx

        self.gamma.data -= lr * self.gamma.grad
        self.beta.data -= lr * self.beta.grad

        return Tensor(dL_dx)

    def get_params(self):
        return {
            'gamma': self.gamma.data,
            'beta': self.beta.data,
            'running_mean': self.running_mean,
            'running_var': self.running_var
        }

    def set_params(self, params):
        self.gamma.data = params['gamma']
        self.beta.data = params['beta']
        self.running_mean = params['running_mean']
        self.running_var = params['running_var']


class InstanceNormLayer:
    """
    Parameters:
    -----------
    self num_features: num_features
    self eps: eps
    self gamma: gamma
    self.beta: beta
    self.running_mean: running_mean
    self.running_var: running_var
    self.training: training

    Explanations:
    ------------
    self.training: If True, the layer is in training mode, else in evaluation mode
    self.gamma and self.beta are learnable parameters
    self.momentum is a hyperparameter
    self.eps is a small value to avoid division by zero
    self.running_mean and self.running_var are running estimates of the mean and variance
    self.x_centered is the centered input
    self.x_norm is the normalized input
    self.outputs is the output of the layer
    self.num_features is the number of features in the input
    """

    def __init__(self, num_features, eps=1e-5):
        self.num_features = num_features
        self.eps = eps
        self.gamma = Tensor(np.ones((1, num_features)))
        self.beta = Tensor(np.zeros((1, num_features)))

    def forward(self, inputs: Tensor):
        self.inputs = inputs
        self.batch_size, _ = inputs.data.shape

        mean = np.mean(inputs.data, axis=1, keepdims=True)
        var = np.var(inputs.data, axis=1, keepdims=True)

        self.x_centered = inputs.data - mean
        self.x_norm = self.x_centered / np.sqrt(var + self.eps)
        outputs = self.gamma.data * self.x_norm + self.beta.data
        self.outputs = Tensor(outputs)
        return self.outputs

    def backward(self, dL_dout: Tensor, lr: float):
        dL_dgamma = np.sum(dL_dout.data * self.x_norm, axis=(0, 1), keepdims=True)
        dL_dbeta = np.sum(dL_dout.data, axis=(0, 1), keepdims=True)

        dL_dx_norm = dL_dout.data * self.gamma.data
        dL_dvar = np.sum(
            dL_dx_norm * self.x_centered * -0.5 * (self.inputs.data.var(axis=1, keepdims=True) + self.eps) ** (-3 / 2),
            axis=1, keepdims=True)
        dL_dmean = np.sum(dL_dx_norm * -1 / np.sqrt(self.inputs.data.var(axis=1, keepdims=True) + self.eps), axis=1,
                          keepdims=True)
        dL_dmean += dL_dvar * np.mean(-2 * self.x_centered, axis=1, keepdims=True)

        dL_dx = dL_dx_norm / np.sqrt(self.inputs.data.var(axis=1, keepdims=True) + self.eps)
        dL_dx += dL_dvar * 2 * self.x_centered / self.inputs.data.shape[1]
        dL_dx += dL_dmean / self.inputs.data.shape[1]

        self.gamma.grad = dL_dgamma if self.gamma.grad is None else self.gamma.grad + dL_dgamma
        self.beta.grad = dL_dbeta if self.beta.grad is None else self.beta.grad + dL_dbeta
        self.inputs.backward_fn = lambda grad: grad + dL_dx if self.inputs.backward_fn is None else lambda \
                x: self.inputs.backward_fn(x) + dL_dx

        self.gamma.data -= lr * self.gamma.grad
        self.beta.data -= lr * self.beta.grad

        return Tensor(dL_dx)

    def get_params(self):
        return {
            'gamma': self.gamma.data,
            'beta': self.beta.data
        }

    def set_params(self, params):
        self.gamma.data = params['gamma']
        self.beta.data = params['beta']


class DropoutLayer:
    """
    Parameters:
    -----------
    self rate: rate
    self.mask: mask
    self.inputs: inputs
    self.outputs: outputs
    self.training: training

    Explanations:
    ------------
    self.training: If True, the layer is in training mode, else in evaluation mode
    self.mask is the dropout mask
    self.outputs is the output of the layer
    """

    def __init__(self, rate=0.5):
        self.rate = rate

    def forward(self, inputs: Tensor, training=True):
        self.inputs = inputs
        if not training:
            return inputs
        self.mask = np.random.binomial(1, 1 - self.rate, size=inputs.data.shape) / (1 - self.rate)
        self.outputs = Tensor(inputs.data * self.mask)
        return self.outputs

    def backward(self, dL_dout: Tensor, lr: float):
        dL_dinputs = dL_dout.data * self.mask

        self.inputs.grad = dL_dinputs if self.inputs.grad is None else self.inputs.grad + dL_dinputs
        self.inputs.backward_fn = lambda grad: grad + dL_dinputs if self.inputs.backward_fn is None else lambda \
                x: self.inputs.backward_fn(x) + dL_dinputs

        return Tensor(dL_dinputs)

    def get_params(self):
        return None

    def set_params(self, params):
        pass


class CrossEntropyLoss:
    """
    Parameters:
    -----------
    self outputs: outputs
    self targets: targets

    """

    def forward(self, outputs: Tensor, targets: Tensor):
        samples = len(outputs.data)
        clipped_outputs = np.clip(outputs.data, 1e-12, 1 - 1e-12)
        correct_confidences = clipped_outputs[range(samples), targets.data.astype(int)]
        negative_log_likelihoods = -np.log(correct_confidences)
        loss = np.mean(negative_log_likelihoods)
        self.outputs = outputs
        self.targets = targets
        return loss

    def backward(self, dL_dloss: Tensor, lr: float):
        samples = len(self.outputs.data)
        clipped_outputs = np.clip(self.outputs.data, 1e-12, 1 - 1e-12)
        clipped_outputs[range(samples), self.targets.data.astype(int)] -= 1
        dL_doutputs = Tensor(clipped_outputs / samples)

        self.outputs.grad = dL_doutputs.data if self.outputs.grad is None else self.outputs.grad + dL_doutputs.data
        self.outputs.backward_fn = lambda grad: grad + dL_doutputs.data if self.outputs.backward_fn is None else lambda \
                x: self.outputs.backward_fn(x) + dL_doutputs.data

        return dL_doutputs


class SGDOptimizer:
    """
    Parameters:
    -----------
    self lr: learning rate
    self momentum: momentum
    self velocity: velocity
    """

    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.velocity = {}

    def update(self, layer, grad, key):
        if key not in self.velocity:
            self.velocity[key] = np.zeros_like(grad)
        self.velocity[key] = self.momentum * self.velocity[key] - self.lr * grad
        layer += self.velocity[key]

        layer.grad = np.zeros_like(grad)
        layer.backward_fn = None


class LSTMLayer:
    """
    Parameters:
    -----------
    self input_size: input size
    self hidden_size: hidden size
    self Wf: weight for forget gate
    self Wi: weight for input gate
    self Wc: weight for cell gate
    self Wo: weight for output gate
    self bf: bias for forget gate
    self bi: bias for input gate
    self bc: bias for cell gate
    self bo: bias for output gate
    self hidden: hidden
    self cell: cell
    self outputs: output
    """

    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.Wf = Tensor(np.random.randn(input_size + hidden_size, hidden_size) * 0.01)
        self.Wi = Tensor(np.random.randn(input_size + hidden_size, hidden_size) * 0.01)
        self.Wc = Tensor(np.random.randn(input_size + hidden_size, hidden_size) * 0.01)
        self.Wo = Tensor(np.random.randn(input_size + hidden_size, hidden_size) * 0.01)

        self.bf = Tensor(np.zeros((1, hidden_size)))
        self.bi = Tensor(np.zeros((1, hidden_size)))
        self.bc = Tensor(np.zeros((1, hidden_size)))
        self.bo = Tensor(np.zeros((1, hidden_size)))

    def forward(self, inputs: Tensor, prev_hidden=None, prev_cell=None):
        if prev_hidden is None:
            prev_hidden = Tensor(np.zeros((inputs.data.shape[0], self.hidden_size)))
        if prev_cell is None:
            prev_cell = Tensor(np.zeros((inputs.data.shape[0], self.hidden_size)))

        self.inputs = inputs
        self.prev_hidden = prev_hidden
        self.prev_cell = prev_cell

        combined = np.concatenate((inputs.data, prev_hidden.data), axis=1)

        f = self.sigmoid(Tensor(np.dot(combined, self.Wf.data) + self.bf.data))

        i = self.sigmoid(Tensor(np.dot(combined, self.Wi.data) + self.bi.data))

        c_candidate = np.tanh(np.dot(combined, self.Wc.data) + self.bc.data)

        cell = f.data * prev_cell.data + i.data * c_candidate

        o = self.sigmoid(Tensor(np.dot(combined, self.Wo.data) + self.bo.data))

        hidden = o.data * np.tanh(cell)

        self.f, self.i, self.c_candidate, self.o = f, i, Tensor(c_candidate), o
        self.cell = Tensor(cell)
        self.hidden = Tensor(hidden)

        return self.hidden, self.cell

    def backward(self, dL_dh: Tensor, dL_dc: Tensor, lr: float):
        dL_do = dL_dh.data * np.tanh(self.cell.data)
        dL_dcell = dL_dc.data + dL_dh.data * self.o.data * (1 - np.tanh(self.cell.data) ** 2)

        dL_df = dL_dcell * self.prev_cell.data
        dL_di = dL_dcell * self.c_candidate.data
        dL_dc_candidate = dL_dcell * self.i.data

        dL_dWf = np.dot(np.concatenate((self.inputs.data, self.prev_hidden.data), axis=1).T,
                        dL_df * self.f.data * (1 - self.f.data))
        dL_dWi = np.dot(np.concatenate((self.inputs.data, self.prev_hidden.data), axis=1).T,
                        dL_di * self.i.data * (1 - self.i.data))
        dL_dWc = np.dot(np.concatenate((self.inputs.data, self.prev_hidden.data), axis=1).T,
                        dL_dc_candidate * (1 - self.c_candidate.data ** 2))
        dL_dWo = np.dot(np.concatenate((self.inputs.data, self.prev_hidden.data), axis=1).T,
                        dL_do * self.o.data * (1 - self.o.data))

        dL_dbf = np.sum(dL_df * self.f.data * (1 - self.f.data), axis=0, keepdims=True)
        dL_dbi = np.sum(dL_di * self.i.data * (1 - self.i.data), axis=0, keepdims=True)
        dL_dbc = np.sum(dL_dc_candidate * (1 - self.c_candidate.data ** 2), axis=0, keepdims=True)
        dL_dbo = np.sum(dL_do * self.o.data * (1 - self.o.data), axis=0, keepdims=True)

        self.Wf.grad = dL_dWf if self.Wf.grad is None else self.Wf.grad + dL_dWf
        self.Wi.grad = dL_dWi if self.Wi.grad is None else self.Wi.grad + dL_dWi
        self.Wc.grad = dL_dWc if self.Wc.grad is None else self.Wc.grad + dL_dWc
        self.Wo.grad = dL_dWo if self.Wo.grad is None else self.Wo.grad + dL_dWo
        self.bf.grad = dL_dbf if self.bf.grad is None else self.bf.grad + dL_dbf
        self.bi.grad = dL_dbi if self.bi.grad is None else self.bi.grad + dL_dbi
        self.bc.grad = dL_dbc if self.bc.grad is None else self.bc.grad + dL_dbc
        self.bo.grad = dL_dbo if self.bo.grad is None else self.bo.grad + dL_dbo

        dL_dprev_h = np.dot(dL_df * self.f.data * (1 - self.f.data), self.Wf.data[self.input_size:].T) + \
                     np.dot(dL_di * self.i.data * (1 - self.i.data), self.Wi.data[self.input_size:].T) + \
                     np.dot(dL_dc_candidate * (1 - self.c_candidate.data ** 2), self.Wc.data[self.input_size:].T) + \
                     np.dot(dL_do * self.o.data * (1 - self.o.data), self.Wo.data[self.input_size:].T)

        dL_dprev_c = dL_dcell * self.f.data

        self.prev_hidden.grad = dL_dprev_h if self.prev_hidden.grad is None else self.prev_hidden.grad + dL_dprev_h
        self.prev_hidden.backward_fn = lambda \
                grad: grad + dL_dprev_h if self.prev_hidden.backward_fn is None else lambda \
            x: self.prev_hidden.backward_fn(
            x) + dL_dprev_h

        self.prev_cell.grad = dL_dprev_c if self.prev_cell.grad is None else self.prev_cell.grad + dL_dprev_c
        self.prev_cell.backward_fn = lambda grad: grad + dL_dprev_c if self.prev_cell.backward_fn is None else lambda \
                x: self.prev_cell.backward_fn(x) + dL_dprev_c

        return Tensor(dL_dprev_h), Tensor(dL_dprev_c)

    def sigmoid(self, x: Tensor):
        return x.apply(lambda z: 1 / (1 + np.exp(-z)))

    def get_params(self):
        return {
            'Wf': self.Wf.data, 'Wi': self.Wi.data, 'Wc': self.Wc.data, 'Wo': self.Wo.data,
            'bf': self.bf.data, 'bi': self.bi.data, 'bc': self.bc.data, 'bo': self.bo.data
        }

    def set_params(self, params):
        self.Wf.data = params['Wf']
        self.Wi.data = params['Wi']
        self.Wc.data = params['Wc']
        self.Wo.data = params['Wo']
        self.bf.data = params['bf']
        self.bi.data = params['bi']
        self.bc.data = params['bc']
        self.bo.data = params['bo']


class GRULayer:
    # Simillar to LTSM layer
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.Wz = Tensor(np.random.randn(input_size + hidden_size, hidden_size) * 0.01)
        self.Wr = Tensor(np.random.randn(input_size + hidden_size, hidden_size) * 0.01)
        self.Wh = Tensor(np.random.randn(input_size + hidden_size, hidden_size) * 0.01)

        self.bz = Tensor(np.zeros((1, hidden_size)))
        self.br = Tensor(np.zeros((1, hidden_size)))
        self.bh = Tensor(np.zeros((1, hidden_size)))

    def forward(self, inputs: Tensor, prev_hidden=None):
        if prev_hidden is None:
            prev_hidden = Tensor(np.zeros((inputs.data.shape[0], self.hidden_size)))

        self.inputs = inputs
        self.prev_hidden = prev_hidden

        combined = np.concatenate((inputs.data, prev_hidden.data), axis=1)

        z = self.sigmoid(Tensor(np.dot(combined, self.Wz.data) + self.bz.data))

        r = self.sigmoid(Tensor(np.dot(combined, self.Wr.data) + self.br.data))

        h_candidate = np.tanh(
            np.dot(np.concatenate((inputs.data, r.data * prev_hidden.data), axis=1), self.Wh.data) + self.bh.data)

        hidden = (1 - z.data) * prev_hidden.data + z.data * h_candidate

        self.z, self.r, self.h_candidate = z, r, Tensor(h_candidate)
        self.hidden = Tensor(hidden)

        return self.hidden

    def backward(self, dL_dh: Tensor, lr: float):
        dL_dz = dL_dh.data * (self.h_candidate.data - self.prev_hidden.data)
        dL_dh_candidate = dL_dh.data * self.z.data
        dL_dr = np.dot(dL_dh_candidate * (1 - self.h_candidate.data ** 2),
                       self.Wh.data[self.input_size:].T) * self.prev_hidden.data

        dL_dWz = np.dot(np.concatenate((self.inputs.data, self.prev_hidden.data), axis=1).T,
                        dL_dz * self.z.data * (1 - self.z.data))
        dL_dWr = np.dot(np.concatenate((self.inputs.data, self.prev_hidden.data), axis=1).T,
                        dL_dr * self.r.data * (1 - self.r.data))
        dL_dWh = np.dot(np.concatenate((self.inputs.data, self.r.data * self.prev_hidden.data), axis=1).T,
                        dL_dh_candidate * (1 - self.h_candidate.data ** 2))

        dL_dbz = np.sum(dL_dz * self.z.data * (1 - self.z.data), axis=0, keepdims=True)
        dL_dbr = np.sum(dL_dr * self.r.data * (1 - self.r.data), axis=0, keepdims=True)
        dL_dbh = np.sum(dL_dh_candidate * (1 - self.h_candidate.data ** 2), axis=0, keepdims=True)

        self.Wz.grad = dL_dWz if self.Wz.grad is None else self.Wz.grad + dL_dWz
        self.Wr.grad = dL_dWr if self.Wr.grad is None else self.Wr.grad + dL_dWr
        self.Wh.grad = dL_dWh if self.Wh.grad is None else self.Wh.grad + dL_dWh
        self.bz.grad = dL_dbz if self.bz.grad is None else self.bz.grad + dL_dbz
        self.br.grad = dL_dbr if self.br.grad is None else self.br.grad + dL_dbr
        self.bh.grad = dL_dbh if self.bh.grad is None else self.bh.grad + dL_dbh

        dL_dprev_h = np.dot(dL_dz * self.z.data * (1 - self.z.data), self.Wz.data[self.input_size:].T) + \
                     np.dot(dL_dr * self.r.data * (1 - self.r.data), self.Wr.data[self.input_size:].T) + \
                     np.dot(dL_dh_candidate * (1 - self.h_candidate.data ** 2),
                            self.Wh.data[self.input_size:].T) * self.r.data + \
                     dL_dh.data * (1 - self.z.data)

        dL_dx = np.dot(dL_dz * self.z.data * (1 - self.z.data), self.Wz.data[:self.input_size].T) + \
                np.dot(dL_dr * self.r.data * (1 - self.r.data), self.Wr.data[:self.input_size].T) + \
                np.dot(dL_dh_candidate * (1 - self.h_candidate.data ** 2), self.Wh.data[:self.input_size].T)

        self.prev_hidden.grad = dL_dprev_h if self.prev_hidden.grad is None else self.prev_hidden.grad + dL_dprev_h
        self.prev_hidden.backward_fn = lambda \
                grad: grad + dL_dprev_h if self.prev_hidden.backward_fn is None else lambda \
            x: self.prev_hidden.backward_fn(
            x) + dL_dprev_h

        self.inputs.grad = dL_dx if self.inputs.grad is None else self.inputs.grad + dL_dx
        self.inputs.backward_fn = lambda grad: grad + dL_dx if self.inputs.backward_fn is None else lambda \
                x: self.inputs.backward_fn(x) + dL_dx

        return Tensor(dL_dx), Tensor(dL_dprev_h)

    def sigmoid(self, x: Tensor):
        return x.apply(lambda z: 1 / (1 + np.exp(-z)))

    def get_params(self):
        return {
            'Wz': self.Wz.data, 'Wr': self.Wr.data, 'Wh': self.Wh.data,
            'bz': self.bz.data, 'br': self.br.data, 'bh': self.bh.data
        }

    def set_params(self, params):
        self.Wz.data = params['Wz']
        self.Wr.data = params['Wr']
        self.Wh.data = params['Wh']
        self.bz.data = params['bz']
        self.br.data = params['br']
        self.bh.data = params['bh']


class RNNLayer:
    # Simmilar to GRULayer
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.Wh = Tensor(np.random.randn(hidden_size, hidden_size) * 0.01)
        self.Wx = Tensor(np.random.randn(input_size, hidden_size) * 0.01)
        self.b = Tensor(np.zeros((1, hidden_size)))

    def forward(self, inputs: Tensor, prev_hidden=None):
        if prev_hidden is None:
            prev_hidden = Tensor(np.zeros((inputs.data.shape[0], self.hidden_size)))

        self.inputs = inputs
        self.prev_hidden = prev_hidden

        hidden = np.tanh(np.dot(inputs.data, self.Wx.data) +
                         np.dot(prev_hidden.data, self.Wh.data) +
                         self.b.data)

        self.hidden = Tensor(hidden)
        return self.hidden

    def backward(self, dL_dh: Tensor, lr: float):
        dL_dWh = np.dot(self.prev_hidden.data.T, dL_dh.data * (1 - self.hidden.data ** 2))
        dL_dWx = np.dot(self.inputs.data.T, dL_dh.data * (1 - self.hidden.data ** 2))
        dL_db = np.sum(dL_dh.data * (1 - self.hidden.data ** 2), axis=0, keepdims=True)

        self.Wh.grad = dL_dWh if self.Wh.grad is None else self.Wh.grad + dL_dWh
        self.Wx.grad = dL_dWx if self.Wx.grad is None else self.Wx.grad + dL_dWx
        self.b.grad = dL_db if self.b.grad is None else self.b.grad + dL_db

        dL_dprev_h = np.dot(dL_dh.data * (1 - self.hidden.data ** 2), self.Wh.data.T)
        dL_dx = np.dot(dL_dh.data * (1 - self.hidden.data ** 2), self.Wx.data.T)

        self.prev_hidden.grad = dL_dprev_h if self.prev_hidden.grad is None else self.prev_hidden.grad + dL_dprev_h
        self.inputs.grad = dL_dx if self.inputs.grad is None else self.inputs.grad + dL_dx

        return Tensor(dL_dx), Tensor(dL_dprev_h)

    def get_params(self):
        return {
            'Wh': self.Wh.data,
            'Wx': self.Wx.data,
            'b': self.b.data
        }

    def set_params(self, params):
        self.Wh.data = params['Wh']
        self.Wx.data = params['Wx']
        self.b.data = params['b']


class DummyLayer:
    # A dummy layer for testing
    def forward(self, inputs):
        return inputs

    def backward(self, grad, lr):
        return grad

    def get_params(self):
        return {}

    def set_params(self, params):
        pass


class MaxPoolingLayer:
    """
    Paramaters:
    self inputs = the inputs
    self output = the output
    self pool_size = the pool size
    self stride = the stride

    Explanation:
    Pooling layer with max pooling
    """

    def __init__(self, pool_size=2, stride=2):
        self.pool_size = pool_size
        self.stride = stride

    def forward(self, inputs):
        batch_size, channels, height, width = inputs.data.shape
        pool_height, pool_width = self.pool_size, self.pool_size
        stride = self.stride

        out_height = (height - pool_height) // stride + 1
        out_width = (width - pool_width) // stride + 1

        output = np.zeros((batch_size, channels, out_height, out_width))

        for i in range(out_height):
            for j in range(out_width):
                h_start = i * stride
                h_end = h_start + pool_height
                w_start = j * stride
                w_end = w_start + pool_width
                output[:, :, i, j] = np.max(inputs.data[:, :, h_start:h_end, w_start:w_end], axis=(2, 3))

        self.inputs = inputs
        self.output = Tensor(output)
        return self.output

    def backward(self, dL_dout: Tensor, lr: float = None):
        dL_dinputs = np.zeros_like(self.inputs.data)

        new_h, new_w, c = dL_dout.data.shape
        for i in range(new_h):
            for j in range(new_w):
                for k in range(c):
                    pool_region = self.inputs.data[i * self.stride:i * self.stride + self.pool_size,
                                  j * self.stride:j * self.stride + self.pool_size, k]
                    max_val = np.max(pool_region)
                    for m in range(self.pool_size):
                        for n in range(self.pool_size):
                            if pool_region[m, n] == max_val:
                                dL_dinputs[i * self.stride + m, j * self.stride + n, k] = dL_dout.data[i, j, k]

        self.inputs.grad = dL_dinputs if self.inputs.grad is None else self.inputs.grad + dL_dinputs
        self.inputs.backward_fn = lambda grad: grad + dL_dinputs if self.inputs.backward_fn is None else lambda \
                x: self.inputs.backward_fn(x) + dL_dinputs

        return Tensor(dL_dinputs)

    def get_params(self):
        return None

    def set_params(self, params):
        pass


class AveragePoolingLayer:
    # Simmilar to MaxPoolingLayer but with avaerage pooling
    def __init__(self, pool_size=2, stride=2):
        self.pool_size = pool_size
        self.stride = stride

    def forward(self, inputs):
        batch_size, channels, height, width = inputs.data.shape
        pool_height, pool_width = self.pool_size, self.pool_size
        stride = self.stride

        out_height = (height - pool_height) // stride + 1
        out_width = (width - pool_width) // stride + 1

        output = np.zeros((batch_size, channels, out_height, out_width))

        for i in range(out_height):
            for j in range(out_width):
                h_start = i * stride
                h_end = h_start + pool_height
                w_start = j * stride
                w_end = w_start + pool_width
                output[:, :, i, j] = np.mean(inputs.data[:, :, h_start:h_end, w_start:w_end], axis=(2, 3))

        self.inputs = inputs
        self.output = Tensor(output)
        return self.output

    def backward(self, dL_dout: Tensor, lr: float = None):
        dL_dinputs = np.zeros_like(self.inputs.data)

        new_h, new_w, c = dL_dout.data.shape
        for i in range(new_h):
            for j in range(new_w):
                for k in range(c):
                    avg_val = dL_dout.data[i, j, k] / (self.pool_size * self.pool_size)
                    for m in range(self.pool_size):
                        for n in range(self.pool_size):
                            dL_dinputs[i * self.stride + m, j * self.stride + n, k] += avg_val

        self.inputs.grad = dL_dinputs if self.inputs.grad is None else self.inputs.grad + dL_dinputs
        self.inputs.backward_fn = lambda grad: grad + dL_dinputs if self.inputs.backward_fn is None else lambda \
                x: self.inputs.backward_fn(x) + dL_dinputs

        return Tensor(dL_dinputs)

    def get_params(self):
        return None

    def set_params(self, params):
        pass


class AdamOptimizer:
    # Generate doctstring codeium
    """
    Parameters
    ----------
    lr : float
        Learning rate
    beta1 : float
        Exponential decay rate for the first moment estimates
    beta2 : float
        Exponential decay rate for the second moment estimates
    epsilon : float
        Constant for numerical stability
    """

    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {}
        self.v = {}
        self.t = 0

    def update(self, layer, grad, key):
        self.t += 1

        if key not in self.m:
            self.m[key] = np.zeros_like(grad)
            self.v[key] = np.zeros_like(grad)

        self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grad
        self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (grad ** 2)

        m_hat = self.m[key] / (1 - self.beta1 ** self.t)
        v_hat = self.v[key] / (1 - self.beta2 ** self.t)

        update = self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
        layer -= update

    def reset(self):
        self.m = {}
        self.v = {}
        self.t = 0


class NadamOptimizer:
    # Simmilar to Adam Optimizer
    def __init__(self, lr=0.002, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {}
        self.v = {}
        self.t = 0

    def update(self, layer, grad, key):
        self.t += 1

        if key not in self.m:
            self.m[key] = np.zeros_like(grad)
            self.v[key] = np.zeros_like(grad)

        beta1_t = self.beta1 * (1 - self.beta1 ** (self.t - 1)) / (1 - self.beta1 ** self.t)
        beta2_t = self.beta2 * (1 - self.beta2 ** (self.t - 1)) / (1 -
                                                                   self.beta2 ** self.t)

        self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grad
        self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (grad ** 2)

        m_hat = self.m[key] / (1 - self.beta1 ** self.t)
        v_hat = self.v[key] / (1 - self.beta2 ** self.t)

        m_nesterov = beta1_t * m_hat + (1 - beta1_t) * grad

        update = self.lr * m_nesterov / (np.sqrt(v_hat) + self.epsilon)

        layer -= update

    def reset(self):
        self.m = {}
        self.v = {}
        self.t = 0


class AdadeltaOptimizer:
    """
    Parameters
    ----------
    rho : float
        Decay rate for the first moment estimates
    epsilon : float
        Constant for numerical stability
    """

    def __init__(self, rho=0.95, epsilon=1e-8):
        self.rho = rho
        self.epsilon = epsilon
        self.E_g2 = {}
        self.E_dx2 = {}

    def update(self, layer, grad, key):
        if key not in self.E_g2:
            self.E_g2[key] = np.zeros_like(grad)
            self.E_dx2[key] = np.zeros_like(grad)

        self.E_g2[key] = self.rho * self.E_g2[key] + (1 - self.rho) * (grad ** 2)

        RMS_g = np.sqrt(self.E_g2[key] + self.epsilon)
        RMS_dx = np.sqrt(self.E_dx2[key] + self.epsilon)
        update = (RMS_dx / RMS_g) * grad

        self.E_dx2[key] = self.rho * self.E_dx2[key] + (1 - self.rho) * (update ** 2)

        layer -= update

    def reset(self):
        self.E_g2 = {}
        self.E_dx2 = {}


class MSELoss:
    # Basic loss function
    def forward(self, outputs, targets):
        return np.mean((outputs.data - targets.data) ** 2)

    def backward(self, outputs, targets):
        return Tensor(2 * (outputs.data - targets.data) / targets.data.size)


class MeanSquaredLogarithmicError:
    """
    Parameters
    ----------
    epsilon : float
        Constant for numerical stability
    """

    def forward(self, outputs: Tensor, targets: Tensor):
        epsilon = 1e-8
        return np.mean((np.log(outputs.data + 1 + epsilon) - np.log(targets.data + 1 + epsilon)) ** 2)

    def backward(self, outputs: Tensor, targets: Tensor):
        epsilon = 1e-8
        grad = 2 * (np.log(outputs.data + 1 + epsilon) - np.log(targets.data + 1 + epsilon)) / (
                outputs.data + 1 + epsilon)
        return Tensor(grad / targets.data.size)


class MeanAbsolutePercentageError:
    """
    Parameters
    ----------
    epsilon : float
        Constant for numerical stability
    """

    def forward(self, outputs: Tensor, targets: Tensor):
        epsilon = 1e-8
        return np.mean(np.abs((targets.data - outputs.data) / (targets.data + epsilon)) * 100)

    def backward(self, outputs: Tensor, targets: Tensor):
        epsilon = 1e-8
        grad = -100 * np.sign(targets.data - outputs.data) / (targets.data + epsilon)
        return Tensor(grad / targets.data.size)


class CosineSimilarityLoss:
    """
    Parameters
    ----------
    dot_product : float
        Dot product of outputs and targets
    outputs_norm : float
        Norm of outputs
    targets_norm : float
        Norm of targets
    cosine_similarity : float
        Cosine similarity of outputs and targets
    """

    def forward(self, outputs: Tensor, targets: Tensor):
        dot_product = np.sum(outputs.data * targets.data, axis=1)
        outputs_norm = np.linalg.norm(outputs.data, axis=1)
        targets_norm = np.linalg.norm(targets.data, axis=1)
        cosine_similarity = dot_product / (outputs_norm * targets_norm + 1e-8)
        return np.mean(1 - cosine_similarity)

    def backward(self, outputs: Tensor, targets: Tensor):
        dot_product = np.sum(outputs.data * targets.data, axis=1, keepdims=True)
        outputs_norm = np.linalg.norm(outputs.data, axis=1, keepdims=True)
        targets_norm = np.linalg.norm(targets.data, axis=1, keepdims=True)

        grad = (targets.data / (outputs_norm * targets_norm + 1e-8)) - \
               (outputs.data * dot_product / (outputs_norm ** 3 * targets_norm + 1e-8))

        return Tensor(-grad / targets.data.shape[0])


class PaddingLayer:
    """
    Parameters:
    -----------
    self.padding : Union[int, Tuple[int, int], Tuple[int, int, int, int]] a padding attribute
    """

    def __init__(self, padding: Union[int, Tuple[int, int], Tuple[int, int, int, int]]):
        if isinstance(padding, int):
            self.padding = ((padding, padding), (padding, padding))
        elif len(padding) == 2:
            self.padding = ((padding[0], padding[0]), (padding[1], padding[1]))
        elif len(padding) == 4:
            self.padding = ((padding[0], padding[1]), (padding[2], padding[3]))
        else:
            raise ValueError(
                "Invalid padding format. Use int, (pad_h, pad_w) or (pad_top, pad_bottom, pad_left, pad_right)")

    def forward(self, inputs: Tensor) -> Tensor:
        raise NotImplementedError

    def backward(self, dL_dout: Tensor, lr: float = None) -> Tensor:
        raise NotImplementedError

    def get_params(self):
        return None

    def set_params(self, params):
        pass


class ZeroPaddingLayer(PaddingLayer):
    # The same like in PaddingLayer but with paddings set to 0
    def forward(self, inputs: Tensor) -> Tensor:
        self.inputs = inputs
        padded = np.pad(inputs.data, ((0, 0), (0, 0)) + self.padding, mode='constant', constant_values=0)
        return Tensor(padded)

    def backward(self, dL_dout: Tensor, lr: float = None) -> Tensor:
        pad_top, pad_bottom = self.padding[0]
        pad_left, pad_right = self.padding[1]
        dL_dinputs = dL_dout.data[:, :, pad_top:-pad_bottom, pad_left:-pad_right]

        self.inputs.grad = dL_dinputs if self.inputs.grad is None else self.inputs.grad + dL_dinputs
        self.inputs.backward_fn = lambda grad: grad + dL_dinputs if self.inputs.backward_fn is None else lambda \
                x: self.inputs.backward_fn(x) + dL_dinputs

        return Tensor(dL_dinputs)


class ReflectionPaddingLayer(PaddingLayer):
    # The same like in PaddingLayer but with paddings reflected
    def forward(self, inputs: Tensor) -> Tensor:
        self.inputs = inputs
        padded = np.pad(inputs.data, ((0, 0), (0, 0)) + self.padding, mode='reflect')
        return Tensor(padded)

    def backward(self, dL_dout: Tensor, lr: float = None) -> Tensor:
        pad_top, pad_bottom = self.padding[0]
        pad_left, pad_right = self.padding[1]
        dL_dinputs = dL_dout.data[:, :, pad_top:-pad_bottom, pad_left:-pad_right]

        self.inputs.grad = dL_dinputs if self.inputs.grad is None else self.inputs.grad + dL_dinputs
        self.inputs.backward_fn = lambda grad: grad + dL_dinputs if self.inputs.backward_fn is None else lambda \
                x: self.inputs.backward_fn(x) + dL_dinputs

        return Tensor(dL_dinputs)


class LinearActivationLayer:
    # Activation layer for linear activation, linear algebras
    def forward(self, inputs: Tensor):
        self.inputs = inputs
        return inputs

    def backward(self, dL_dout: Tensor, lr: float):
        dL_dinputs = dL_dout.data

        self.inputs.grad = dL_dinputs if self.inputs.grad is None else self.inputs.grad + dL_dinputs
        self.inputs.backward_fn = lambda grad: grad + dL_dinputs if self.inputs.backward_fn is None else lambda \
                x: self.inputs.backward_fn(x) + dL_dinputs

        return Tensor(dL_dinputs)

    def get_params(self):
        return None

    def set_params(self, params):
        pass


class BilinearLayer:
    """
    Parameters:
    -----------
    in1_features : int
    in2_features : int
    out_features : int
    self.weight : Tensor
    self.bias : Tensor
    Explanations :
    -------------
    in1_features: input size of the first input
    in2_features: input size of the second input
    out_features: output size
    self.weight: weight matrix
    self.bias: bias
    """

    def __init__(self, in1_features, in2_features, out_features):
        self.in1_features = in1_features
        self.in2_features = in2_features
        self.out_features = out_features

        self.weight = Tensor(
            np.random.randn(out_features, in1_features, in2_features) / np.sqrt(in1_features * in2_features))
        self.bias = Tensor(np.zeros(out_features))

    def forward(self, input1: Tensor, input2: Tensor):
        self.input1 = input1
        self.input2 = input2
        output = np.einsum('bi,bj,oij->bo', input1.data, input2.data, self.weight.data) + self.bias.data
        return Tensor(output)

    def backward(self, dL_dout: Tensor, lr: float):
        dL_dw = np.einsum('bo,bi,bj->oij', dL_dout.data, self.input1.data, self.input2.data)
        dL_db = np.sum(dL_dout.data, axis=0)
        dL_dinput1 = np.einsum('bo,oij,bj->bi', dL_dout.data, self.weight.data, self.input2.data)
        dL_dinput2 = np.einsum('bo,oij,bi->bj', dL_dout.data, self.weight.data, self.input1.data)

        self.weight.data -= lr * dL_dw
        self.bias.data -= lr * dL_db

        self.input1.grad = dL_dinput1 if self.input1.grad is None else self.input1.grad + dL_dinput1
        self.input2.grad = dL_dinput2 if self.input2.grad is None else self.input2.grad + dL_dinput2

        self.input1.backward_fn = lambda grad: grad + dL_dinput1 if self.input1.backward_fn is None else lambda \
                x: self.input1.backward_fn(x) + dL_dinput1
        self.input2.backward_fn = lambda grad: grad + dL_dinput2 if self.input2.backward_fn is None else lambda \
                x: self.input2.backward_fn(x) + dL_dinput2

        return Tensor(dL_dinput1), Tensor(dL_dinput2)

    def get_params(self):
        return [self.weight, self.bias]

    def set_params(self, params):
        self.weight, self.bias = params


class TanhActivationLayer:
    """
    Parameters:
    -----------
    inputs : Tensor
    output : Tensor
    dl_dinputs : Tensor
    dl_doutputs : Tensor
    Explanations :
    -------------
    inputs: input tensor
    output: output tensor
    dl_dinputs: gradient of the loss with respect to the inputs
    dl_doutputs: gradient of the loss with respect to the outputs
    """

    def forward(self, inputs: Tensor):
        self.inputs = inputs
        self.output = Tensor(np.tanh(inputs.data))
        return self.output

    def backward(self, dL_dout: Tensor, lr: float):
        dL_dinputs = dL_dout.data * (1 - np.tanh(self.inputs.data) ** 2)

        if self.inputs.requires_grad:
            self.inputs.grad = dL_dinputs if self.inputs.grad is None else self.inputs.grad + dL_dinputs
            if self.inputs.backward_fn is None:
                self.inputs.backward_fn = lambda grad: grad + dL_dinputs
            else:
                old_backward_fn = self.inputs.backward_fn
                self.inputs.backward_fn = lambda grad: old_backward_fn(grad) + dL_dinputs

        return Tensor(dL_dinputs)

    def get_params(self):
        return None

    def set_params(self, params):
        pass


class HardTanhActivationLayer:
    """
    Parameters:
    -----------
    inputs : Tensor
    output : Tensor
    dl_dinputs : Tensor
    dl_doutputs : Tensor
    Explanations :
    -------------
    inputs: input tensor
    output: output tensor
    dl_dinputs: gradient of the loss with respect to the inputs
    dl_doutputs: gradient of the loss with respect to the outputs
    """

    def __init__(self):
        self.inputs: Optional[Tensor] = None
        self.output: Optional[Tensor] = None
        self.dl_dinputs: Optional[Tensor] = None
        self.dl_doutputs: Optional[Tensor] = None

    def forward(self, inputs: Tensor) -> Tensor:
        self.inputs = inputs
        self.output = Tensor(np.clip(inputs.data, -1, 1))
        return self.output

    def backward(self, dL_dout: Tensor, lr: float) -> Tensor:
        dL_dinputs = dL_dout.data * np.logical_and(self.inputs.data >= -1, self.inputs.data <= 1).astype(float)
        if self.inputs.requires_grad:
            self.inputs.grad = dL_dinputs if self.inputs.grad is None else self.inputs.grad + dL_dinputs
            if self.inputs.backward_fn is None:
                self.inputs.backward_fn = lambda grad: grad + dL_dinputs
            else:
                old_backward_fn = self.inputs.backward_fn
                self.inputs.backward_fn = lambda grad: old_backward_fn(grad) + dL_dinputs
        return Tensor(dL_dinputs)

    def get_params(self) -> None:
        return None

    def set_params(self, params) -> None:
        pass


class ScaledDotProductAttention:
    """
    Parameters:
    -----------
    d_model : int
    attention_scores : Tensor
    attention_weights : Tensor
    Explanations :
    -------------
    d_model: dimension of the model
    attention_scores: attention scores
    attention_weights: attention weights
    """

    def __init__(self, d_model):
        self.d_model = d_model
        self.scale = np.sqrt(d_model)

    def forward(self, Q: Tensor, K: Tensor, V: Tensor, mask=None):
        self.Q, self.K, self.V = Q, K, V

        attention_scores = Q.dot(K.transpose())
        attention_scores.data /= self.scale

        if mask is not None:
            attention_scores.data += (mask.data * -1e9)

        attention_weights = self.softmax(attention_scores)
        self.attention_weights = attention_weights

        output = attention_weights.dot(V)
        return output

    def backward(self, dL_dout: Tensor, lr: float):
        dL_dV = self.attention_weights.transpose().dot(dL_dout)
        dL_dattention_weights = dL_dout.dot(self.V.transpose())
        dL_dattention_scores = dL_dattention_weights * (self.attention_weights * (1 - self.attention_weights.data))
        dL_dattention_scores = dL_dattention_scores * (1 / self.scale)
        dL_dQ = dL_dattention_scores.dot(self.K)
        dL_dK = dL_dattention_scores.transpose().dot(self.Q)

        self.Q.grad = dL_dQ.data if self.Q.grad is None else self.Q.grad + dL_dQ.data
        self.K.grad = dL_dK.data if self.K.grad is None else self.K.grad + dL_dK.data
        self.V.grad = dL_dV.data if self.V.grad is None else self.V.grad + dL_dV.data

        return dL_dQ, dL_dK, dL_dV

    def softmax(self, x: Tensor):
        exp_x = np.exp(x.data - np.max(x.data, axis=-1, keepdims=True))
        return Tensor(exp_x / np.sum(exp_x, axis=-1, keepdims=True))

    def get_params(self):
        return None

    def set_params(self, params):
        pass


class Embedding:
    """
    Parameters:
    -----------
    vocab_size : int
    embedding_dim : int
    embeddings : Tensor
    input_indices : Tensor
    Explanations :
    -------------
    vocab_size: size of the vocabulary
    embedding_dim: dimension of the embedding
    embeddings: embedding matrix
    input_indices: input indices
    """

    def __init__(self, vocab_size: int, embedding_dim: int):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.embeddings = Tensor(np.random.randn(vocab_size, embedding_dim) * 0.01)
        self.input_indices = None

    def forward(self, input_indices: Tensor) -> Tensor:
        self.input_indices = input_indices
        return Tensor(self.embeddings.data[input_indices.data])

    def backward(self, output_gradient: Tensor, lr: float) -> None:
        embedding_gradient = np.zeros_like(self.embeddings.data)
        np.add.at(embedding_gradient, self.input_indices.data, output_gradient.data)
        self.embeddings.data -= lr * embedding_gradient

    def get_params(self) -> Tuple[np.ndarray]:
        return (self.embeddings.data,)

    def set_params(self, params: Tuple[np.ndarray]) -> None:
        self.embeddings.data = params[0]


class PairwiseDistance:
    def __init__(self, p=2, eps=1e-6, keepdim=False):
        """
        Initialize the PairwiseDistance layer.

        Args:
        p (int): The norm degree for pairwise distance. Default is 2 (Euclidean distance).
        eps (float): Small value to avoid division by zero. Default is 1e-6.
        keepdim (bool): Whether to keep the same dimensions as input. Default is False.
        """
        self.p = p
        self.eps = eps
        self.keepdim = keepdim
        self.diff = None
        self.norm = None
        self.x1 = None
        self.x2 = None

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        """
        Compute the pairwise distances between x1 and x2.

        Args:
        x1 (Tensor): First input tensor of shape (N, D)
        x2 (Tensor): Second input tensor of shape (M, D)

        Returns:
        Tensor: Pairwise distances of shape (N, M) or (N, M, 1) if keepdim is True
        """
        self.x1 = x1
        self.x2 = x2
        self.diff = x1.data[:, None, :] - x2.data[None, :, :]
        self.norm = np.power(np.abs(self.diff) + self.eps, self.p)
        output = np.power(np.sum(self.norm, axis=-1), 1 / self.p)

        if self.keepdim:
            output = np.expand_dims(output, -1)

        return Tensor(output)

    def backward(self, grad_output: Tensor, lr: float) -> Tuple[Tensor, Tensor]:
        """
        Compute the gradient of the pairwise distance.

        Args:
        grad_output (Tensor): Gradient of the loss with respect to the output of this layer
        lr (float): Learning rate (not used in this layer, but kept for consistency)

        Returns:
        Tuple[Tensor, Tensor]: Gradients with respect to x1 and x2
        """
        if self.keepdim:
            grad_output_data = grad_output.data.squeeze(-1)
        else:
            grad_output_data = grad_output.data

        grad_output_expanded = grad_output_data[:, :, None]
        dist = np.power(np.sum(self.norm, axis=-1), 1 / self.p - 1)
        dist_expanded = dist[:, :, None]

        grad_dist = self.p * np.power(np.abs(self.diff) + self.eps, self.p - 1) * np.sign(self.diff)
        grad = grad_output_expanded * dist_expanded * grad_dist / self.norm.sum(axis=-1, keepdims=True)

        grad_x1 = grad.sum(axis=1)
        grad_x2 = -grad.sum(axis=0)

        return Tensor(grad_x1), Tensor(grad_x2)

    def get_params(self) -> Tuple:
        return tuple()

    def set_params(self, params: Tuple) -> None:
        pass


class Encoder:
    """
    Params:
    input_dim: int
    hidden_dims: List[int]
    output_dim: int
    """

    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int):
        self.layers = []
        dims = [input_dim] + hidden_dims + [output_dim]

        for i in range(len(dims) - 1):
            self.layers.append({
                'W': Tensor(np.random.randn(dims[i], dims[i + 1]) * 0.01),
                'b': Tensor(np.zeros((1, dims[i + 1])))
            })

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers[:-1]:
            x = x.dot(layer['W']).__add__(layer['b']).apply(lambda x: np.maximum(0, x))

        # Last layer without activation
        x = x.dot(self.layers[-1]['W']).__add__(self.layers[-1]['b'])
        return x

    def parameters(self) -> List[Tensor]:
        return [param for layer in self.layers for param in layer.values()]


class Decoder:
    """
    Params:
    input_dim: int
    hidden_dims: List[int]
    output_dim: int
    Explanations:
    input_dim: input dimension of the data
    hidden_dims: list of hidden dimensions
    output_dim: output dimension of the data
    """

    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int):
        self.layers = []
        dims = [input_dim] + hidden_dims + [output_dim]
        for i in range(len(dims) - 1):
            self.layers.append({
                'W': Tensor(np.random.randn(dims[i], dims[i + 1]) * 0.01),
                'b': Tensor(np.zeros((1, dims[i + 1])))
            })

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers[:-1]:
            x = x.dot(layer['W']).__add__(layer['b']).apply(lambda x: np.maximum(0, x))  # ReLU activation
        # Last layer with sigmoid activation for output between 0 and 1
        x = x.dot(self.layers[-1]['W']).__add__(self.layers[-1]['b']).apply(lambda x: 1 / (1 + np.exp(-x)))
        return x

    def parameters(self) -> List[Tensor]:
        return [param for layer in self.layers for param in layer.values()]


class TransformerEncoder:
    """
    Params:
    input_dim: int
    num_heads: int
    ff_dim: int
    num_layers: int
    """

    def __init__(self, input_dim: int, num_heads: int, ff_dim: int, num_layers: int):
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.num_layers = num_layers

        self.layers = []
        for _ in range(num_layers):
            self.layers.append({
                'attention': MultiHeadAttention(input_dim, num_heads),
                'norm1': LayerNorm(input_dim),
                'ff': FeedForward(input_dim, ff_dim),
                'norm2': LayerNorm(input_dim)
            })

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            # Self-attention
            attention_output = layer['attention'].forward(x, x, x)
            x = x + attention_output
            x = layer['norm1'].forward(x)

            # Feed-forward
            ff_output = layer['ff'].forward(x)
            x = x + ff_output
            x = layer['norm2'].forward(x)

        return x

    def parameters(self) -> List[Tensor]:
        return [param for layer in self.layers for module in layer.values() for param in module.parameters()]


class TransformerDecoder:
    """
    Params:
    input_dim: int
    num_heads: int
    ff_dim: int
    num_layers: int
    output_dim: int
    """

    def __init__(self, input_dim: int, num_heads: int, ff_dim: int, num_layers: int, output_dim: int):
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.num_layers = num_layers
        self.output_dim = output_dim

        self.layers = []
        for _ in range(num_layers):
            self.layers.append({
                'self_attention': MultiHeadAttention(input_dim, num_heads),
                'norm1': LayerNorm(input_dim),
                'cross_attention': MultiHeadAttention(input_dim, num_heads),
                'norm2': LayerNorm(input_dim),
                'ff': FeedForward(input_dim, ff_dim),
                'norm3': LayerNorm(input_dim)
            })

        self.output_layer = Linear(input_dim, output_dim)

    def forward(self, x: Tensor, encoder_output: Tensor) -> Tensor:
        for layer in self.layers:
            # Self-attention
            self_attention_output = layer['self_attention'].forward(x, x, x)
            x = x + self_attention_output
            x = layer['norm1'].forward(x)

            # Cross-attention
            cross_attention_output = layer['cross_attention'].forward(x, encoder_output, encoder_output)
            x = x + cross_attention_output
            x = layer['norm2'].forward(x)

            # Feed-forward
            ff_output = layer['ff'].forward(x)
            x = x + ff_output
            x = layer['norm3'].forward(x)

        return self.output_layer.forward(x)

    def parameters(self) -> List[Tensor]:
        params = [param for layer in self.layers for module in layer.values() for param in module.parameters()]
        params.extend(self.output_layer.parameters())
        return params


class MultiHeadAttention:
    """
    Params:
    input_dim: int
    num_heads: int
    head_dim = input_dim // num_heads
    Explanations:
    - input_dim: Dimensionality of the input tensor
    - num_heads: Number of attention heads
    - head_dim: Dimensionality of each attention head
    """

    def __init__(self, input_dim: int, num_heads: int):
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads

        # Initialize weights
        self.W_q = Tensor(np.random.randn(input_dim, input_dim) * np.sqrt(2.0 / (2 * input_dim)))
        self.W_k = Tensor(np.random.randn(input_dim, input_dim) * np.sqrt(2.0 / (2 * input_dim)))
        self.W_v = Tensor(np.random.randn(input_dim, input_dim) * np.sqrt(2.0 / (2 * input_dim)))
        self.W_o = Tensor(np.random.randn(input_dim, input_dim) * np.sqrt(2.0 / (2 * input_dim)))

    def split_heads(self, x: np.ndarray) -> np.ndarray:
        batch_size, seq_length, _ = x.shape
        return x.reshape(batch_size, seq_length, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

    def forward(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        batch_size, seq_length, _ = query.data.shape

        # Compute Q, K, V
        q = self.split_heads(np.dot(query.data, self.W_q.data))
        k = self.split_heads(np.dot(key.data, self.W_k.data))
        v = self.split_heads(np.dot(value.data, self.W_v.data))

        # Compute attention scores
        attention_scores = np.matmul(q, k.transpose(0, 1, 3, 2)) / np.sqrt(self.head_dim)

        # Compute attention probabilities
        attention_probs = np.exp(attention_scores - np.max(attention_scores, axis=-1, keepdims=True))
        attention_probs /= np.sum(attention_probs, axis=-1, keepdims=True)

        # Compute context
        context = np.matmul(attention_probs, v)

        # Reshape and apply final linear transformation
        context = context.transpose(0, 2, 1, 3).reshape(batch_size, seq_length, self.input_dim)
        output = np.dot(context, self.W_o.data)

        return Tensor(output)

    def parameters(self) -> List[Tensor]:
        return [self.W_q, self.W_k, self.W_v, self.W_o]


class FeedForward:
    """
    Params:
    W1 = np.random.randn(input_dim, hidden_dim) * 0.01
    b1 = np.zeros((1, hidden_dim))
    W2 = np.random.randn(hidden_dim, input_dim) * 0.01
    b2 = np.zeros((1, input_dim))
    Explanations:
    - W1: Weight matrix for the first linear transformation
    - b1: Bias vector for the first linear transformation
    - W2: Weight matrix for the second linear transformation
    - b2: Bias vector for the second linear transformation
    """

    def __init__(self, input_dim: int, hidden_dim: int):
        self.W1 = Tensor(np.random.randn(input_dim, hidden_dim) * 0.01)
        self.b1 = Tensor(np.zeros((1, hidden_dim)))
        self.W2 = Tensor(np.random.randn(hidden_dim, input_dim) * 0.01)
        self.b2 = Tensor(np.zeros((1, input_dim)))

    def forward(self, x: Tensor) -> Tensor:
        hidden = x.dot(self.W1).__add__(self.b1).apply(lambda x: np.maximum(0, x))
        return hidden.dot(self.W2).__add__(self.b2)

    def parameters(self) -> List[Tensor]:
        return [self.W1, self.b1, self.W2, self.b2]


class LayerNorm:
    """
    Params:
    dim: int
    gamma = np.ones((1, dim))
    beta = np.zeros((1, dim))
    eps = 1e-5
    Explanations:
    - dim: Dimensionality of the input tensor
    - gamma: Scale parameter
    - beta: Shift parameter
    - eps: Epsilon value
    """

    def __init__(self, dim: int):
        self.gamma = Tensor(np.ones((1, dim)))
        self.beta = Tensor(np.zeros((1, dim)))
        self.eps = 1e-5

    def forward(self, x: Tensor) -> Tensor:
        mean = np.mean(x.data, axis=-1, keepdims=True)
        var = np.var(x.data, axis=-1, keepdims=True)
        x_norm = (x.data - mean) / np.sqrt(var + self.eps)
        return Tensor(x_norm * self.gamma.data + self.beta.data)

    def parameters(self) -> List[Tensor]:
        return [self.gamma, self.beta]


class Linear:
    """
    Params:
    - input_dim: Dimensionality of the input tensor
    - output_dim: Dimensionality of the output tensor
    Explanations:
    - input_dim: Dimensionality of the input tensor
    - output_dim: Dimensionality of the output tensor
    """

    def __init__(self, input_dim: int, output_dim: int):
        self.W = Tensor(np.random.randn(input_dim, output_dim) * 0.01)
        self.b = Tensor(np.zeros((1, output_dim)))

    def forward(self, x: Tensor) -> Tensor:
        return x.dot(self.W).__add__(self.b)

    def parameters(self) -> List[Tensor]:
        return [self.W, self.b]


class MaxUnpoolingLayer:
    def __init__(self, pool_size=2, stride=2):
        self.pool_size = pool_size
        self.stride = stride

    def forward(self, inputs):
        batch_size, channels, height, width = inputs.data.shape
        unpooled_height = (height - 1) * self.stride + self.pool_size
        unpooled_width = (width - 1) * self.stride + self.pool_size

        output = np.zeros((batch_size, channels, unpooled_height, unpooled_width))

        for i in range(height):
            for j in range(width):
                h_start = i * self.stride
                w_start = j * self.stride
                output[:, :, h_start:h_start+self.pool_size, w_start:w_start+self.pool_size] = np.expand_dims(inputs.data[:, :, i, j], axis=(2, 3))

        self.inputs = inputs
        self.output = Tensor(output)
        return self.output


    def backward(self, dL_dout: Tensor, lr: float = None):
        dL_dinputs = np.zeros_like(self.inputs.data)

        batch_size, channels, height, width = self.inputs.data.shape
        for i in range(height):
            for j in range(width):
                h_start = i * self.stride
                w_start = j * self.stride
                dL_dinputs[:, :, i, j] = np.sum(dL_dout.data[:, :, h_start:h_start+self.pool_size, w_start:w_start+self.pool_size], axis=(2, 3))

        self.inputs.grad = dL_dinputs if self.inputs.grad is None else self.inputs.grad + dL_dinputs
        self.inputs.backward_fn = lambda grad: grad + dL_dinputs if self.inputs.backward_fn is None else lambda x: self.inputs.backward_fn(x) + dL_dinputs

        return Tensor(dL_dinputs)

    def get_params(self):
        return None

    def set_params(self, params):
        pass

class AverageUnpoolingLayer:
    def __init__(self, pool_size=2, stride=2):
        self.pool_size = pool_size
        self.stride = stride

    def forward(self, inputs):
        batch_size, channels, height, width = inputs.data.shape
        unpooled_height = (height - 1) * self.stride + self.pool_size
        unpooled_width = (width - 1) * self.stride + self.pool_size

        output = np.zeros((batch_size, channels, unpooled_height, unpooled_width))

        for i in range(height):
            for j in range(width):
                h_start = i * self.stride
                w_start = j * self.stride
                output[:, :, h_start:h_start+self.pool_size, w_start:w_start+self.pool_size] = np.expand_dims(inputs.data[:, :, i, j], axis=(2, 3)) / (self.pool_size * self.pool_size)

        self.inputs = inputs
        self.output = Tensor(output)
        return self.output

    def backward(self, dL_dout: Tensor, lr: float = None):
        dL_dinputs = np.zeros_like(self.inputs.data)

        batch_size, channels, height, width = self.inputs.data.shape
        for i in range(height):
            for j in range(width):
                h_start = i * self.stride
                w_start = j * self.stride
                dL_dinputs[:, :, i, j] = np.sum(dL_dout.data[:, :, h_start:h_start+self.pool_size, w_start:w_start+self.pool_size], axis=(2, 3)) / (self.pool_size * self.pool_size)

        self.inputs.grad = dL_dinputs if self.inputs.grad is None else self.inputs.grad + dL_dinputs
        self.inputs.backward_fn = lambda grad: grad + dL_dinputs if self.inputs.backward_fn is None else lambda x: self.inputs.backward_fn(x) + dL_dinputs

        return Tensor(dL_dinputs)

    def get_params(self):
        return None

    def set_params(self, params):
        pass


class NeuralNetwork:
    def __init__(self, temperature=1.0):
        self.layers = []
        self.hooks = {
            'pre_forward': [],
            'post_forward': [],
            'pre_backward': [],
            'post_backward': [],
            'pre_epoch': [],
            'post_epoch': []
        }
        self.temperature = temperature
        self.profiler = cProfile.Profile()
        self.is_profiling = False

    def add(self, layer):
        self.layers.append(layer)

    def remove(self, index):
        del self.layers[index]

    def add_hook(self, hook_type: str, hook_fn: Callable):
        if hook_type in self.hooks:
            self.hooks[hook_type].append(hook_fn)
        else:
            raise ValueError(f"Invalid hook type: {hook_type}")

    def remove_hook(self, hook_type: str, hook_fn: Callable):
        if hook_type in self.hooks and hook_fn in self.hooks[hook_type]:
            self.hooks[hook_type].remove(hook_fn)

    def _run_hooks(self, hook_type: str, *args, **kwargs):
        for hook in self.hooks[hook_type]:
            hook(*args, **kwargs)

    def start_profiling(self):
        """Start profiling."""
        self.profiler.enable()
        self.is_profiling = True

    def stop_profiling(self):
        """Stop profiling."""
        self.profiler.disable()
        self.is_profiling = False

    def print_profile_stats(self, sort_by='cumulative', lines=20):
        """Print profiling statistics."""
        if not self.is_profiling:
            print("Profiling was not started. Use start_profiling() first.")
            return

        s = io.StringIO()
        ps = pstats.Stats(self.profiler, stream=s).sort_stats(sort_by)
        ps.print_stats(lines)
        print(s.getvalue())

    def forward(self, inputs: Tensor):
        if self.is_profiling:
            return self._profiled_forward(inputs)
        else:
            return self._forward(inputs)

    def _profiled_forward(self, inputs: Tensor):
        self.profiler.enable()
        result = self._forward(inputs)
        self.profiler.disable()
        return result

    def _forward(self, inputs: Tensor):
        self._run_hooks('pre_forward', inputs)
        for layer in self.layers:
            inputs = layer.forward(inputs)
        # Apply temperature scaling to the final layer output
        inputs.data = inputs.data / self.temperature
        self._run_hooks('post_forward', inputs)
        return inputs

    def backward(self, loss_gradient: Tensor, lr: float):
        if self.is_profiling:
            return self._profiled_backward(loss_gradient, lr)
        else:
            return self._backward(loss_gradient, lr)

    def _profiled_backward(self, loss_gradient: Tensor, lr: float):
        self.profiler.enable()
        result = self._backward(loss_gradient, lr)
        self.profiler.disable()
        return result

    def _backward(self, loss_gradient: Tensor, lr: float):
        self._run_hooks('pre_backward', loss_gradient, lr)
        for layer in reversed(self.layers):
            loss_gradient = layer.backward(loss_gradient, lr)
        self._run_hooks('post_backward', loss_gradient, lr)

    def train(self, inputs: Tensor, targets: Tensor, epochs: int, lr: float, batch_size: int, loss_function):
        if self.is_profiling:
            return self._profiled_train(inputs, targets, epochs, lr, batch_size, loss_function)
        else:
            return self._train(inputs, targets, epochs, lr, batch_size, loss_function)

    def _profiled_train(self, inputs: Tensor, targets: Tensor, epochs: int, lr: float, batch_size: int, loss_function):
        self.profiler.enable()
        result = self._train(inputs, targets, epochs, lr, batch_size, loss_function)
        self.profiler.disable()
        return result

    def _train(self, inputs: Tensor, targets: Tensor, epochs: int, lr: float, batch_size: int, loss_function):
        num_batches = int(np.ceil(inputs.data.shape[0] / batch_size))
        losses = []
        for epoch in range(epochs):
            self._run_hooks('pre_epoch', epoch, epochs)
            epoch_loss = 0
            for batch in range(num_batches):
                batch_start = batch * batch_size
                batch_end = batch_start + batch_size
                batch_inputs = Tensor(inputs.data[batch_start:batch_end])
                batch_targets = Tensor(targets.data[batch_start:batch_end])
                outputs = self.forward(batch_inputs)
                loss = loss_function.forward(outputs, batch_targets)
                epoch_loss += loss
                loss_gradient = loss_function.backward(outputs, batch_targets)
                self.backward(loss_gradient, lr)
            avg_loss = epoch_loss / num_batches
            losses.append(avg_loss)
            self._run_hooks('post_epoch', epoch, epochs, avg_loss)
        return losses


    def save(self, file_path: str):
        params = [layer.get_params() for layer in self.layers]
        with open(file_path, 'wb') as f:
            pickle.dump((params, self.temperature), f)

    def load(self, file_path: str):
        with open(file_path, 'rb') as f:
            params, self.temperature = pickle.load(f)
        for layer, param in zip(self.layers, params):
            layer.set_params(param)

    def set_temperature(self, temperature: float):
        self.temperature = temperature

    def plot_loss(self, losses, title="Training Loss", xlabel="Epoch", ylabel="Loss"):
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(losses) + 1), losses)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.show()
