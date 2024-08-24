import unittest
from okrolearn.okrolearn import Tensor, np


class TestTensorExtensions(unittest.TestCase):

    def test_standardize(self):
        data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        tensor = Tensor(data)
        standardized = tensor.standardize()

        self.assertTrue(np.allclose(np.mean(standardized.data, axis=0), 0, atol=1e-7))
        self.assertTrue(np.allclose(np.std(standardized.data, axis=0), 1, atol=1e-7))

    def test_normalize(self):
        data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        tensor = Tensor(data)
        normalized = tensor.normalize()

        self.assertTrue(np.allclose(np.mean(normalized.data, axis=0), 0, atol=1e-7))
        self.assertTrue(np.allclose(np.std(normalized.data, axis=0), 1, atol=1e-7))

    def test_pca(self):
        data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
        tensor = Tensor(data)
        n_components = 2
        pca_result = tensor.pca(n_components)

        self.assertEqual(pca_result.data.shape, (4, n_components))

    def test_kmeans(self):
        data = np.array([[1, 2], [1, 4], [1, 0], [4, 2], [4, 4], [4, 0]])
        tensor = Tensor(data)
        n_clusters = 2
        labels, centroids = tensor.kmeans(n_clusters)

        self.assertEqual(labels.data.shape, (6,))
        self.assertEqual(centroids.data.shape, (2, 2))
        self.assertEqual(len(np.unique(labels.data)), n_clusters)

    def test_kmeans_backward(self):
        data = np.array([[1, 2], [1, 4], [1, 0], [4, 2], [4, 4], [4, 0]])
        tensor = Tensor(data)
        n_clusters = 2
        labels, _ = tensor.kmeans(n_clusters)

        # Manually trigger backward pass
        labels.backward()

        self.assertIsNotNone(tensor.grad)
        self.assertEqual(tensor.grad.shape, data.shape)

    def test_linear_regression(self):
        X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
        y = np.dot(X, np.array([1, 2])) + 3

        tensor_X = Tensor(X)
        tensor_y = Tensor(y)

        coeffs = tensor_X.linear_regression(tensor_y)

        self.assertEqual(coeffs.data.shape, (3,))  # 2 features + 1 intercept
        self.assertTrue(np.allclose(coeffs.data, [3, 1, 2], atol=1e-7))

    def test_backward_propagation(self):
        # Test if backward propagation works through multiple operations
        data = np.random.rand(100, 2)
        d = Tensor(data)

        e = d.standardize()

        e.backward()

        self.assertIsNotNone(d.grad)


if __name__ == '__main__':
    unittest.main()