from scipy import sparse
import pandas as pd
from okrolearn.okrolearn import np, plt

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
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

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
            plt.plot(self.data.get())
        elif self.data.ndim == 2:
            plt.imshow(self.data.get(), cmap='viridis')
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
        h_data = self.data.get()
        plt.hist(h_data.flatten(), bins=bins)

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

    def copy(self, requires_grad=None):
        """
        Create a copy of the Tensor.

        Parameters:
        - requires_grad: bool, optional. If provided, sets the requires_grad flag for the new Tensor.
                         If not provided, uses the current Tensor's requires_grad value.

        Returns:
        - Tensor: A new Tensor with copied data.
        """
        if requires_grad is None:
            requires_grad = self.requires_grad

        new_tensor = Tensor(np.copy(self.data), requires_grad=requires_grad)

        if self.grad is not None:
            new_tensor.grad = np.copy(self.grad)

        # Note: We don't copy the backward_fn as it's specific to the computation graph

        return new_tensor

    def __repr__(self):
        return f"Tensor({self.data})"

    def backward(self):
        if self.backward_fn is None:
            raise RuntimeError("No backward function defined for this tensor.")
        self.backward_fn(np.ones_like(self.data))
