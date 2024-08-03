from okrolearn.okrolearn import Tensor, np

# Test for polynomial_features
def test_polynomial_features():
    data = np.array([[1, 2], [3, 4]])
    tensor = Tensor(data)
    poly_tensor = tensor.polynomial_features(degree=2)

    expected_output = np.array([[1, 1, 2, 1, 2, 4],
                                [1, 3, 4, 9, 12, 16]])

    assert np.allclose(poly_tensor.data, expected_output), f"Expected {expected_output}, but got {poly_tensor.data}"

# Run the test
test_polynomial_features()
print("Test passed.")
