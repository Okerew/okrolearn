import unittest
from okrolearn.okrolearn import NeuralNetwork, DenseLayer, ReLUActivationLayer, SoftmaxActivationLayer, np

class TestNeuralNetwork(unittest.TestCase):
    def setUp(self):
        self.network = NeuralNetwork()
        self.network.add(DenseLayer(10, 64))
        self.network.add(ReLUActivationLayer())
        self.network.add(DenseLayer(64, 32))
        self.network.add(ReLUActivationLayer())
        self.network.add(DenseLayer(32, 3))
        self.network.add(SoftmaxActivationLayer())

    def generate_toy_data(self, n_samples=100, n_features=10, n_classes=3):
        X = np.random.randn(n_samples, n_features)
        y = np.random.randint(0, n_classes, size=n_samples)
        return X, y

    def test_rank_network_performance_shapes(self):
        X_test, y_test = self.generate_toy_data()

        # Ensure X_test is 2D and y_test is 1D
        self.assertEqual(X_test.ndim, 2)
        self.assertEqual(y_test.ndim, 1)

        temperatures = [0.5, 1.0, 1.5]
        for temp in temperatures:
            result = self.network.rank_network_performance(
                X_test, y_test[:, np.newaxis],  # Convert y_test to 2D array
                temperature=temp,
                task_type='classification',
                creativity_threshold=0.3
            )

            # Check that all expected keys are present
            expected_keys = ['temperature', 'loss', 'accuracy', 'output_diversity',
                             'creativity_alignment', 'performance_score', 'rank', 'execution_time_seconds', 'memory_usage_mb']
            self.assertSetEqual(set(result.keys()), set(expected_keys))

            # Check types and shapes of returned values
            self.assertIsInstance(result['temperature'], float)
            self.assertIsInstance(result['loss'], float)
            self.assertIsInstance(result['accuracy'], float)
            self.assertIsInstance(result['output_diversity'], float)
            self.assertIsInstance(result['creativity_alignment'], float)
            self.assertIsInstance(result['performance_score'], float)
            self.assertIsInstance(result['rank'], str)
            self.assertIsInstance(result['execution_time_seconds'], float)
            self.assertIsInstance(result['memory_usage_mb'], float)

            # Check value ranges
            self.assertGreaterEqual(result['accuracy'], 0.0)
            self.assertLessEqual(result['accuracy'], 1.0)
            self.assertGreaterEqual(result['creativity_alignment'], 0.0)
            self.assertLessEqual(result['creativity_alignment'], 1.0)
            self.assertGreaterEqual(result['performance_score'], 0.0)
            self.assertLessEqual(result['performance_score'], 1.0)

            # Check rank is one of the expected values
            self.assertIn(result['rank'], ['Excellent', 'Good', 'Fair', 'Poor'])

    def test_rank_network_performance_temperature_effect(self):
        X_test, y_test = self.generate_toy_data()

        # Ensure X_test is 2D and y_test is 1D
        self.assertEqual(X_test.ndim, 2)
        self.assertEqual(y_test.ndim, 1)

        results = []
        temperatures = [0.5, 1.0, 1.5]
        for temp in temperatures:
            result = self.network.rank_network_performance(
                X_test, y_test[:, np.newaxis],  # Convert y_test to 2D array
                temperature=temp,
                task_type='classification',
                creativity_threshold=0.3
            )
            results.append(result)

        # Check that output diversity increases with temperature
        self.assertLessEqual(results[0]['output_diversity'], results[1]['output_diversity'])
        self.assertLessEqual(results[1]['output_diversity'], results[2]['output_diversity'])

    def test_rank_network_performance_error_handling(self):
        X_test, y_test = self.generate_toy_data()

        # Ensure X_test is 2D and y_test is 1D
        self.assertEqual(X_test.ndim, 2)
        self.assertEqual(y_test.ndim, 1)

        # Test with invalid task type
        with self.assertRaises(ValueError):
            self.network.rank_network_performance(
                X_test, y_test,
                temperature=1.0,
                task_type='invalid_task',
                creativity_threshold=0.3
            )

        # Test with mismatched input shapes
        with self.assertRaises(ValueError):
            self.network.rank_network_performance(
                X_test[:, :5], y_test[:5],  # Using only half the features
                temperature=1.0,
                task_type='classification',
                creativity_threshold=0.3
            )

if __name__ == '__main__':
    unittest.main()
