import numpy as np
from typing import Tuple
from src.okrolearn.okrolearn import *

def test_conv3d_layer():
    # Initialize layer
    in_channels, out_channels = 3, 2
    kernel_size = (3, 3, 3)
    conv3d = Conv3DLayer(in_channels, out_channels, kernel_size, stride=1, padding=0)
    
    # Set deterministic weights and biases for testing
    conv3d.filters.data = np.random.randn(out_channels, in_channels, *kernel_size).astype(np.float32) * 0.1
    conv3d.biases.data = np.zeros((out_channels, 1), dtype=np.float32)
    
    # Create input
    input_shape: Tuple[int, int, int, int] = (1, in_channels, 5, 5, 5) 
    input_data = np.random.randn(*input_shape).astype(np.float32)
    inputs = Tensor(input_data)
    
    # Forward pass
    output = conv3d.forward(inputs)
    
    # Print output for debugging
    print("Conv3D Output:")
    print(output.data)
    
    # Calculate expected output (for demonstration, you should calculate it based on your implementation)
    # This is a simplified expected output
    expected_output = np.random.randn(1, out_channels, 3, 3, 3).astype(np.float32)
    
    # Print expected output for comparison
    print("Expected Output:")
    print(expected_output)
    
    # Check if forward pass is correct with a tolerance
    if np.allclose(output.data, expected_output, atol=1e-3):
        print("Conv3D forward pass succeeded!")
    else:
        print("Conv3D forward pass failed. Difference:")
        print(output.data - expected_output)
    

# Run the test
test_conv3d_layer()

