from okrolearn.okrolearn import *
from okrolearn.optimizers import AdamOptimizer
network = NeuralNetwork()
network.add(DenseLayer(3, 4))
network.add(ReLUActivationLayer())
network.add(DenseLayer(4, 4))
network.add(ReLUActivationLayer())
network.add(DenseLayer(4, 3))
network.add(SoftmaxActivationLayer())
inputs = Tensor(np.random.rand(100, 3))
targets = Tensor(np.random.randint(0, 3, size=(100,)))
loss_function = CrossEntropyLoss()
optimizer = AdamOptimizer(lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8)
network.train(inputs, targets, epochs=1000, lr=0.01, optimizer=optimizer, batch_size=10, loss_function=loss_function)
network.save('model.pt')
test_network = NeuralNetwork()
test_network.add(DenseLayer(3, 4))
test_network.add(ReLUActivationLayer())
test_network.add(DenseLayer(4, 4))
test_network.add(ReLUActivationLayer())
test_network.add(DenseLayer(4, 3))
test_network.add(SoftmaxActivationLayer())
test_network.load('model.pt')
test_inputs = Tensor(np.random.rand(10, 3))
test_outputs = test_network.forward(test_inputs)
print(test_outputs)

