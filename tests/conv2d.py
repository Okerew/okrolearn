import numpy as np
from src.okrolearn.okrolearn import *

# Assuming the Conv2DLayer and Tensor classes are defined as in your provided code

def test_conv2d_layer():
    # Initialize layer
    in_channels, out_channels, kernel_size = 3, 2, (3, 3)
    conv2d = Conv2DLayer(in_channels, out_channels, kernel_size, stride=1, padding=0)
    
    # Set deterministic weights and biases for testing
    conv2d.filters.data = np.array([
        [[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
         [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
         [[1, 2, 3], [4, 5, 6], [7, 8, 9]]],
        [[[9, 8, 7], [6, 5, 4], [3, 2, 1]],
         [[9, 8, 7], [6, 5, 4], [3, 2, 1]],
         [[9, 8, 7], [6, 5, 4], [3, 2, 1]]]
    ])
    conv2d.biases.data = np.array([[1], [2]])
    
    # Create input
    input_data = np.array([
        [[[ 1,  2,  3,  4],
          [ 5,  6,  7,  8],
          [ 9, 10, 11, 12],
          [13, 14, 15, 16]],
         [[17, 18, 19, 20],
          [21, 22, 23, 24],
          [25, 26, 27, 28],
          [29, 30, 31, 32]],
         [[33, 34, 35, 36],
          [37, 38, 39, 40],
          [41, 42, 43, 44],
          [45, 46, 47, 48]]]
    ])
    inputs = Tensor(input_data)
    
    # Forward pass
    output = conv2d.forward(inputs)
    print(output.data)
    
    # Expected output (calculated manually)
    expected_output = np.array([
        [[[1428, 1531],
          [1840, 1943]],
         [[1053, 948],
          [633, 528]]]
    ])
    
    # Check if forward pass is correct
    assert np.allclose(output.data, expected_output), "Conv2D forward pass failed"
    
    # Backward pass
    dL_dout = Tensor(np.ones_like(output.data))
    dL_dinputs = conv2d.backward(dL_dout, lr=0.01)
    
    # Check if backward pass produces gradients
    assert conv2d.filters.grad is not None, "Conv2D filters gradients not computed"
    assert conv2d.biases.grad is not None, "Conv2D biases gradients not computed"
    assert dL_dinputs.data.shape == input_data.shape, "Conv2D input gradients shape mismatch"
    
    print("Conv2DLayer test passed!")

# Run the test
test_conv2d_layer()
