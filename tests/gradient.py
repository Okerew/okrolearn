from okrolearn.okrolearn import NeuralNetwork, Tensor, np
network = NeuralNetwork()
network.set_debug_mode(True)

# Forward pass
inputs = Tensor(np.random.randn(10, 5))
outputs = network.forward(inputs)

# Backward pass
loss_gradient = Tensor(np.random.randn(10, 5))
# Backward pass
network.backward(loss_gradient, lr=0.01)

# Access gradients
for layer_index, gradient in network.gradients:
    print(f"Layer {layer_index} Gradient: {gradient}")