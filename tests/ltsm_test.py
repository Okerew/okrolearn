from okrolearn.src.okrolearn.okrolearn import *

# Create a simple network with a single LSTM layer
class SimpleNetwork:
    def __init__(self, input_size, hidden_size):
        self.lstm = LSTMLayer(input_size, hidden_size)
    
    def forward(self, inputs):
        hidden = None
        cell = None
        outputs = []
        for t in range(inputs.data.shape[1]):
            hidden, cell = self.lstm.forward(Tensor(inputs.data[:, t, :]), hidden, cell)
            outputs.append(hidden.data)
        return np.array(outputs)
    
    def save(self, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump(self.lstm.get_params(), f)
    
    def load(self, file_path):
        with open(file_path, 'rb') as f:
            params = pickle.load(f)
        self.lstm.set_params(params)

# Create and test the network
input_size = 3
hidden_size = 4
sequence_length = 5

# Create the network
network = SimpleNetwork(input_size, hidden_size)

# Generate sample input data
inputs = Tensor(np.random.rand(10, sequence_length, input_size))

# Forward pass
outputs = network.forward(inputs)
print("Output shape:", outputs.shape)
print("Sample output:\n", outputs[0])

# Save the model
network.save('lstm_model.pt')

# Create a new network and load the saved model
test_network = SimpleNetwork(input_size, hidden_size)
test_network.load('lstm_model.pt')

# Test the loaded network
test_inputs = Tensor(np.random.rand(5, sequence_length, input_size))
test_outputs = test_network.forward(test_inputs)
print("\nTest output shape:", test_outputs.shape)
print("Sample test output:\n", test_outputs[0])
