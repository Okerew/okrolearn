from src.okrolearn.okrolearn import NeuralNetwork, Tensor, PairwiseDistance, MSELoss, np

def test_pairwise_distance():
    # Test initialization
    pd = PairwiseDistance(p=2, eps=1e-6, keepdim=True)
    assert pd.p == 2
    assert pd.eps == 1e-6
    assert pd.keepdim == True

    # Test forward pass
    x1 = Tensor(np.array([[1, 2, 3], [4, 5, 6]], dtype=float))
    x2 = Tensor(np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]], dtype=float))
    output = pd.forward(x1, x2)

    print(output.data)
    
    expected_output = np.array([
        [[np.sqrt(5)], [np.sqrt(14)], [np.sqrt(29)]],
        [[np.sqrt(50)], [np.sqrt(35)], [np.sqrt(26)]]
    ])

    # Test backward pass
    grad_output = Tensor(np.ones_like(output.data))
    grad_x1, grad_x2 = pd.backward(grad_output, lr=0.01)
    
    # Check shapes of gradients
    print(grad_x1.data.shape, grad_x2.data.shape)

    # Test integration with NeuralNetwork
    class CustomLayer:
        def forward(self, x1, x2):
            return pd.forward(x1, x2)
        
        def backward(self, grad_output, lr):
            return pd.backward(grad_output, lr)
        
        def get_params(self):
            return tuple()
        
        def set_params(self, params):
            pass

    nn = NeuralNetwork()
    nn.add(CustomLayer())
    
    # Train the network
    inputs1 = Tensor(np.random.randn(100, 3))
    inputs2 = Tensor(np.random.randn(50, 3))
    targets = Tensor(np.random.randn(100, 50, 1))
    
    def custom_forward(nn, inputs):
        return nn.forward(inputs1, inputs2)
    
    nn.forward = lambda inputs: custom_forward(nn, inputs)
    
    nn.train(inputs1, targets, epochs=5, lr=0.01, batch_size=32, loss_function=MSELoss())

# Run the test
test_pairwise_distance()
