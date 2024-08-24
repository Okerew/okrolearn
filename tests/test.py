from okrolearn.okrolearn import *
from okrolearn.optimizers import SGDOptimizer

def print_epoch_start(epoch, total_epochs):
    print(f"Starting epoch {epoch + 1}/{total_epochs}")


network = NeuralNetwork(temperature=0.5)
network.start_profiling()
network.set_debug_mode(True)
network.set_breakpoint('forward', lambda inputs: np.any(np.isnan(inputs.data)))
network.set_breakpoint('backward', lambda grad, lr: np.max(np.abs(grad.data)) > 10)
network.add(DenseLayer(3, 4))
network.add_hook('pre_epoch', print_epoch_start)
network.add(ReLUActivationLayer())
network.add(DenseLayer(4, 4))
network.add(LinearActivationLayer())
network.analyze_gradients()
network.add(LeakyReLUActivationLayer(alpha=0.1))
network.add(DenseLayer(4, 3))
network.add(ELUActivationLayer())
network.add(SoftsignActivationLayer())
network.add(HardTanhActivationLayer())
network.remove(2)
network.add(SoftmaxActivationLayer())
network.add(ELUActivationLayer())
network.add(SELUActivationLayer(alpha=0.1))
network.add(GELUActivationLayer())
inputs = Tensor(np.random.rand(100, 3))
targets = Tensor(np.random.randint(0, 3, size=(100,)))
loss_function = CrossEntropyLoss()
optimizer = SGDOptimizer(lr=0.01)

losses = network.train(inputs, targets, epochs=50, lr=0.01, optimizer=optimizer, batch_size=10, loss_function=loss_function, )
network.plot_parameter_changes()
network.analyze_gradients()
network.print_profile_stats()
network.stop_profiling()

# Plot the training loss
network.plot_loss(losses)
network.save_weights('model.pt')

test_network = NeuralNetwork()
test_network.set_debug_mode(True)
test_network.add(DenseLayer(3, 4))
test_network.add_hook('pre_epoch', print_epoch_start)
test_network.add(ReLUActivationLayer())
test_network.analyze_gradients()
test_network.add(DenseLayer(4, 4))
test_network.add(LinearActivationLayer())
test_network.add(LeakyReLUActivationLayer(alpha=0.1))
test_network.add(DenseLayer(4, 3))
test_network.add(ELUActivationLayer())
test_network.add(SoftsignActivationLayer())
test_network.add(HardTanhActivationLayer())
test_network.remove(2)
test_network.add(SoftmaxActivationLayer())
test_network.add(ELUActivationLayer())
test_network.add(SELUActivationLayer(alpha=0.1))
test_network.add(GELUActivationLayer())

test_network.load_weights('model.pt')
test_network.train(inputs, targets, epochs=50, lr=0.01, optimizer=optimizer, batch_size=10, loss_function=loss_function)
test_network.save("model2.pt")

test_inputs = Tensor(np.random.rand(10, 3))
test_outputs = test_network.forward(test_inputs)
print(test_outputs)

# Visualize a test input
test_inputs.plot(title="Test Input", xlabel="Feature", ylabel="Value")

# Visualize test output distribution
test_outputs.histogram(title="Test Output Distribution", xlabel="Output Value", ylabel="Frequency")
