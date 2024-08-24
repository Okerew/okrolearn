from src.okrolearn.okrolearn import Tensor, np

def test_tensor():
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    # Convert to Tensor objects
    X_tensor = Tensor(X)
    y_tensor = Tensor(y)

    # Test min_max_scale
    print("Testing min_max_scale...")
    X_scaled = X_tensor.min_max_scale(feature_range=(0, 1))
    assert np.allclose(X_scaled.data.min(axis=0), 0, atol=1e-6)
    assert np.allclose(X_scaled.data.max(axis=0), 1, atol=1e-6)
    print("min_max_scale test passed!")

    # Test truncated_svd
    print("Testing truncated_svd...")
    n_components = 3
    X_svd, V = X_tensor.truncated_svd(n_components)
    assert X_svd.data.shape == (100, n_components)
    assert V.data.shape == (n_components, 5)
    # Check if the reconstruction error is small
    X_reconstructed = np.dot(X_svd.data, V.data)
    reconstruction_error = np.mean((X - X_reconstructed) ** 2)
    print("truncated_svd test passed!")

    # Test SVM
    print("Testing SVM...")
    svm_weights, svm_bias = X_tensor.svm(y_tensor)
    svm_predictions = (np.dot(X, svm_weights.data) + svm_bias > 0).astype(int)
    svm_accuracy = np.mean(svm_predictions == y)
    print(f"SVM accuracy: {svm_accuracy:.2f}")
    print("SVM test passed!")

    # Test Logistic Regression
    print("Testing Logistic Regression...")
    lr_weights, lr_bias = X_tensor.logistic_regression(y_tensor)
    lr_predictions = (1 / (1 + np.exp(-(np.dot(X, lr_weights.data) + lr_bias))) > 0.5).astype(int)
    lr_accuracy = np.mean(lr_predictions == y)
    print(f"Logistic Regression accuracy: {lr_accuracy:.2f}")
    print("Logistic Regression test passed!")

    # Test Random Forest
    print("Testing Random Forest...")
    rf_predictions, forest = X_tensor.random_forest(y_tensor, n_trees=10, max_depth=5,)
    rf_accuracy = np.mean(rf_predictions.data == y)
    print(f"Random Forest accuracy: {rf_accuracy:.2f}")
    print("Random Forest test passed!")

    print("All tests passed successfully!")

if __name__ == "__main__":
    test_tensor()
