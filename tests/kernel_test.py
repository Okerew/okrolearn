from src.okrolearn.okrolearn import NeuralNetwork, np
nn = NeuralNetwork()

# Create a custom kernel
kernel_code = r'''
extern "C" __global__
void my_custom_kernel(float* input, float* output, int n) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < n) {
        output[tid] = input[tid] * 2;
    }
}
'''
nn.create_custom_kernel("my_custom_kernel", kernel_code)

# Use the custom kernel
input_data = np.array([1, 2, 3, 4, 5], dtype=np.float32)
output_data = np.zeros_like(input_data)
nn.run_custom_kernel("my_custom_kernel", (1,), (5,), (input_data, output_data, 5))


# Print the results
print("Input data:", input_data)
print("Output data:", output_data)

# Verify the results
expected_output = input_data * 2
np.testing.assert_array_almost_equal(output_data, expected_output)
print("Custom kernel test passed!")