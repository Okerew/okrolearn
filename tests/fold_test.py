from src.okrolearn.okrolearn import Tensor, Fold, Unfold, np
def test_fold_unfold():
    # Test parameters
    batch_size = 2
    channels = 3
    height = 6
    width = 6
    kernel_size = 2
    stride = 2
    padding = 0

    # Create a sample input tensor
    input_data = np.arange(batch_size * channels * height * width).reshape(batch_size, channels, height, width)
    input_tensor = Tensor(input_data)

    # Test Unfold
    unfold = Unfold(kernel_size, stride, padding)
    unfolded = unfold.forward(input_tensor)

    # Check unfolded shape
    expected_unfold_shape = (batch_size, channels * kernel_size * kernel_size, 9)
    assert unfolded.data.shape == expected_unfold_shape, f"Unfold shape mismatch. Expected {expected_unfold_shape}, got {unfolded.data.shape}"

    # Test Fold
    fold = Fold((height, width), kernel_size, stride, padding)
    folded = fold.forward(unfolded)

    # Check folded shape
    expected_fold_shape = (batch_size, channels, height, width)
    assert folded.data.shape == expected_fold_shape, f"Fold shape mismatch. Expected {expected_fold_shape}, got {folded.data.shape}"

    # Check if fold(unfold(x)) ≈ x
    np.testing.assert_allclose(folded.data, input_data, atol=1e-6, 
                               err_msg="Fold(Unfold(x)) should approximately equal x")

    # Test backward pass
    dL_dout = Tensor(np.random.randn(*folded.data.shape))
    
    # Backward through Fold
    dL_dunfolded = fold.backward(dL_dout)
    assert dL_dunfolded.data.shape == unfolded.data.shape, "Fold backward shape mismatch"

    # Backward through Unfold
    dL_dinput = unfold.backward(dL_dunfolded)
    assert dL_dinput.data.shape == input_data.shape, "Unfold backward shape mismatch"

    print("All fold/unfold tests passed successfully!")

# Run the test
if __name__ == "__main__":
    test_fold_unfold()
