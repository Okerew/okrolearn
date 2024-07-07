from okrolearn.src.okrolearn import Tensor, np, plt
from statsmodels.tsa.stattools import acf, pacf
from scipy import stats


class DataAnalyzer:
    def __init__(self, data: Tensor):
        self.data = data.data  # Access the underlying numpy array

    def describe(self):
        """
        Provide basic statistical description of the data.
        """
        return {
            'mean': np.mean(self.data, axis=0),
            'std': np.std(self.data, axis=0),
            'min': np.min(self.data, axis=0),
            'max': np.max(self.data, axis=0),
            'median': np.median(self.data, axis=0),
            'skewness': self.skewness(),
            'kurtosis': self.kurtosis(),
            'shape': self.data.shape
        }

    def skewness(self):
        """
        Compute the skewness of the data.
        """
        n = self.data.shape[0]
        m3 = np.mean((self.data - np.mean(self.data, axis=0)) ** 3, axis=0)
        m2 = np.mean((self.data - np.mean(self.data, axis=0)) ** 2, axis=0)
        return m3 / (m2 ** 1.5)

    def kurtosis(self):
        """
        Compute the kurtosis of the data.
        """
        n = self.data.shape[0]
        m4 = np.mean((self.data - np.mean(self.data, axis=0)) ** 4, axis=0)
        m2 = np.mean((self.data - np.mean(self.data, axis=0)) ** 2, axis=0)
        return m4 / (m2 ** 2) - 3

    def correlation_matrix(self):
        """
        Compute the correlation matrix of the data.
        """
        return np.corrcoef(self.data.T)

    def plot_correlation_heatmap(self):
        """
        Plot a heatmap of the correlation matrix.
        """
        corr = self.correlation_matrix()
        plt.figure(figsize=(10, 8))
        plt.imshow(corr, cmap='coolwarm')
        plt.colorbar()
        plt.title('Correlation Heatmap')
        plt.show()

    def pca_analysis(self, n_components=2):
        """
        Perform PCA analysis on the data.
        """
        # Center the data
        centered_data = self.data - np.mean(self.data, axis=0)

        # Compute covariance matrix
        cov_matrix = np.cov(centered_data.T)

        # Compute eigenvectors and eigenvalues
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

        # Sort eigenvectors by decreasing eigenvalues
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Select top n_components eigenvectors
        components = eigenvectors[:, :n_components]

        # Project data onto principal components
        pca_result = np.dot(centered_data, components)

        # Compute explained variance ratio
        explained_variance_ratio = eigenvalues[:n_components] / np.sum(eigenvalues)

        return pca_result, explained_variance_ratio

    def plot_pca(self, n_components=2):
        """
        Plot the first two principal components.
        """
        pca_result, explained_var = self.pca_analysis(n_components)
        plt.figure(figsize=(10, 8))
        plt.scatter(pca_result[:, 0], pca_result[:, 1])
        plt.xlabel(f'PC1 ({explained_var[0]:.2f})')
        plt.ylabel(f'PC2 ({explained_var[1]:.2f})')
        plt.title('PCA of the data')
        plt.show()

    def kmeans_clustering(self, n_clusters=3, max_iter=100):
        """
        Perform K-means clustering on the data.
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

        return labels, centroids

    def plot_kmeans(self, n_clusters=3):
        """
        Plot K-means clustering results (works best with 2D data).
        """
        labels, centers = self.kmeans_clustering(n_clusters)
        plt.figure(figsize=(10, 8))
        plt.scatter(self.data[:, 0], self.data[:, 1], c=labels, cmap='viridis')
        plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='x', s=200, linewidths=3)
        plt.title('K-means Clustering')
        plt.show()

    def histogram(self, bins=50):
        """
        Plot histograms for each feature in the data.
        """
        n_features = self.data.shape[1]
        fig, axes = plt.subplots(n_features, 1, figsize=(10, 4 * n_features))
        for i in range(n_features):
            if n_features > 1:
                ax = axes[i]
            else:
                ax = axes
            ax.hist(self.data[:, i], bins=bins)
            ax.set_title(f'Feature {i + 1} Distribution')
        plt.tight_layout()
        plt.show()

    def scatter_matrix(self):
        """
        Create a scatter plot matrix of the data.
        """
        n_features = self.data.shape[1]
        fig, axes = plt.subplots(n_features, n_features, figsize=(3 * n_features, 3 * n_features))
        for i in range(n_features):
            for j in range(n_features):
                if i != j:
                    axes[i, j].scatter(self.data[:, j], self.data[:, i], alpha=0.5)
                else:
                    axes[i, j].hist(self.data[:, i])
                if i == n_features - 1:
                    axes[i, j].set_xlabel(f'Feature {j + 1}')
                if j == 0:
                    axes[i, j].set_ylabel(f'Feature {i + 1}')
        plt.tight_layout()
        plt.show()

    def outlier_detection(self, threshold=3):
        """
        Detect outliers using the Z-score method.
        """
        z_scores = (self.data - np.mean(self.data, axis=0)) / np.std(self.data, axis=0)
        outliers = np.abs(z_scores) > threshold
        return outliers

    def plot_outliers(self, threshold=3):
        """
        Plot outliers for each feature.
        """
        outliers = self.outlier_detection(threshold)
        n_features = self.data.shape[1]
        fig, axes = plt.subplots(n_features, 1, figsize=(10, 4 * n_features))
        for i in range(n_features):
            if n_features > 1:
                ax = axes[i]
            else:
                ax = axes
            ax.scatter(range(len(self.data)), self.data[:, i], c='blue', alpha=0.5)
            ax.scatter(np.where(outliers[:, i])[0], self.data[outliers[:, i], i], c='red')
            ax.set_title(f'Feature {i + 1} Outliers')
        plt.tight_layout()
        plt.show()

    def feature_importance(self, target):
        """
        Calculate feature importance using correlation with target variable.
        """
        correlations = np.abs(np.corrcoef(self.data.T, target)[:-1, -1])
        return correlations

    def plot_feature_importance(self, target):
        """
        Plot feature importance.
        """
        importance = self.feature_importance(target)
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(importance)), importance)
        plt.title('Feature Importance')
        plt.xlabel('Feature Index')
        plt.ylabel('Importance (Absolute Correlation)')
        plt.show()

    def analyze_time_series(self, feature_index=0, lags=20):
        """
        Analyze time series patterns for a specific feature.
        """
        data = self.data[:, feature_index]

        # Autocorrelation and Partial Autocorrelation
        acf_values = acf(data, nlags=lags)
        pacf_values = pacf(data, nlags=lags)

        # Plot ACF and PACF
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
        ax1.plot(range(lags + 1), acf_values)
        ax1.set_title('Autocorrelation Function')
        ax2.plot(range(lags + 1), pacf_values)
        ax2.set_title('Partial Autocorrelation Function')
        plt.tight_layout()
        plt.show()

    def detect_seasonality(self, feature_index=0):
        """
        Detect seasonality in a specific feature using FFT.
        """
        data = self.data[:, feature_index]
        fft = np.fft.fft(data)
        frequencies = np.fft.fftfreq(len(data))

        # Plot the power spectrum
        plt.figure(figsize=(10, 6))
        plt.plot(frequencies, np.abs(fft))
        plt.title('Power Spectrum')
        plt.xlabel('Frequency')
        plt.ylabel('Magnitude')
        plt.show()

    def trend_analysis(self, feature_index=0):
        """
        Perform trend analysis on a specific feature.
        """
        data = self.data[:, feature_index]
        x = np.arange(len(data))

        # Linear trend
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, data)

        plt.figure(figsize=(10, 6))
        plt.scatter(x, data, label='Data')
        plt.plot(x, slope * x + intercept, color='red', label='Linear Trend')
        plt.title('Trend Analysis')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend()
        plt.show()

    def analyze_patterns(self, feature_index=0):
        """
        Comprehensive pattern analysis for a specific feature using Matplotlib.
        """
        data = self.data[:, feature_index]

        # Create a 2x2 grid of subplots
        fig, axs = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle(f'Pattern Analysis for Feature {feature_index}', fontsize=16)

        # Time Series Plot
        axs[0, 0].plot(data)
        axs[0, 0].set_title('Time Series')
        axs[0, 0].set_xlabel('Time')
        axs[0, 0].set_ylabel('Value')
        axs[0, 0].grid(True)

        lags = 20
        acf_values = acf(data, nlags=lags)
        axs[0, 1].bar(range(lags + 1), acf_values)
        axs[0, 1].set_title('Autocorrelation Function')
        axs[0, 1].set_xlabel('Lag')
        axs[0, 1].set_ylabel('Correlation')
        axs[0, 1].grid(True)

        fft = np.fft.fft(data)
        frequencies = np.fft.fftfreq(len(data))
        axs[1, 0].plot(frequencies, np.abs(fft))
        axs[1, 0].set_title('Power Spectrum')
        axs[1, 0].set_xlabel('Frequency')
        axs[1, 0].set_ylabel('Magnitude')
        axs[1, 0].set_xlim(0, max(frequencies))
        axs[1, 0].grid(True)

        # Trend Analysis
        x = np.arange(len(data))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, data)
        axs[1, 1].scatter(x, data, alpha=0.5, label='Data')
        axs[1, 1].plot(x, slope * x + intercept, color='red', label='Linear Trend')
        axs[1, 1].set_title('Trend Analysis')
        axs[1, 1].set_xlabel('Time')
        axs[1, 1].set_ylabel('Value')
        axs[1, 1].legend()
        axs[1, 1].grid(True)

        plt.tight_layout()
        plt.show()