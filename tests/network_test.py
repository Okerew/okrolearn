from okrolearn.okrolearn import *


def print_epoch_start(epoch, total_epochs):
    print(f"Starting epoch {epoch + 1}/{total_epochs}")


network = NeuralNetwork(temperature=0.5)
network.set_breakpoint('forward', lambda inputs: np.any(np.isnan(inputs.data)))
network.set_breakpoint('backward', lambda grad, lr: np.max(np.abs(grad.data)) > 10)
network.add(DenseLayer(3, 4))
network.add_hook('pre_epoch', print_epoch_start)
network.add(ReLUActivationLayer())
network.add(DenseLayer(4, 4))
network.add(LinearActivationLayer())
network.add(LeakyReLUActivationLayer(alpha=0.1))
network.add(DenseLayer(4, 3))
network.add(ELUActivationLayer())
network.add(SoftsignActivationLayer())
network.add(HardTanhActivationLayer())
network.add(SoftmaxActivationLayer())
network.add(ELUActivationLayer())
network.set_loss_function(CrossEntropyLoss())
network.deploy(host='0.0.0.0', port=3000)