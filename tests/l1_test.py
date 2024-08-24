from okrolearn.src.okrolearn.okrolearn import *

def test_batchnorm_l1_layer():
    np.random.seed(0)  # For reproducibility

    # Create dummy input data
    input_data = np.random.randn(5, 10)  # Batch size of 5, 10 features
    inputs = Tensor(input_data)

    # Initialize BatchNormLayer and L1RegularizationLayer
    batch_norm_layer = BatchNormLayer(num_features=10)
    l1_layer = L1RegularizationLayer(layer=batch_norm_layer, lambda_=0.01)

    # Forward pass
    outputs = l1_layer.forward(inputs)

    # Expected forward output shape
    assert outputs.data.shape == (5, 10), "Forward output shape mismatch"

    # Create dummy gradient for backward pass
    grad_output = np.random.randn(5, 10)
    dL_dout = Tensor(grad_output)

    # Learning rate
    lr = 0.01

    # Backward pass
    grads = l1_layer.backward(dL_dout, lr)

    # Expected backward output shape
    assert grads.data.shape == (5, 10), "Backward output shape mismatch"

    # Check that the gamma and beta gradients are not None
    assert batch_norm_layer.gamma.grad is not None, "Gamma gradient is None"
    assert batch_norm_layer.beta.grad is not None, "Beta gradient is None"

    # Check that gamma and beta have been updated
    updated_gamma = batch_norm_layer.gamma.data
    updated_beta = batch_norm_layer.beta.data

    assert not np.allclose(updated_gamma, np.ones((1, 10))), "Gamma not updated"
    assert not np.allclose(updated_beta, np.zeros((1, 10))), "Beta not updated"

    print("All tests passed.")

# Run the test
test_batchnorm_l1_layer()

