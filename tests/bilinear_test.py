from okrolearn.src.okrolearn.okrolearn import BilinearLayer, Tensor, np
# Create a BilinearLayer
bilinear_layer = BilinearLayer(5, 5, 3)

# Create some input tensors
input1 = Tensor(np.random.randn(10, 5))
input2 = Tensor(np.random.randn(10, 5))

# Forward pass
output = bilinear_layer.forward(input1, input2)

# Check the output shape
assert output.data.shape == (10, 3), "Output shape is incorrect"

# Backward pass
dL_dout = Tensor(np.random.randn(10, 3))
lr = 0.01
dL_dinput1, dL_dinput2 = bilinear_layer.backward(dL_dout, lr)

# Check the gradients shape
assert dL_dinput1.data.shape == (10, 5), "Gradient shape for input1 is incorrect"
assert dL_dinput2.data.shape == (10, 5), "Gradient shape for input2 is incorrect"

# Check if the weights and biases have been updated
assert not np.allclose(bilinear_layer.weight.data, np.random.randn(3, 5, 5) / np.sqrt(5 * 5)), "Weights have not been updated"
assert not np.allclose(bilinear_layer.bias.data, np.zeros(3)), "Biases have not been updated"

print("All tests passed!")

