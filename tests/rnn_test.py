from okrolearn.src.okrolearn.okrolearn import *

class RNNNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.rnn = RNNLayer(input_size, hidden_size)
        self.output_layer = DenseLayer(hidden_size, output_size)
    
    def forward(self, inputs):
        hidden = None
        rnn_outputs = []
        for t in range(inputs.data.shape[1]):
            hidden = self.rnn.forward(Tensor(inputs.data[:, t, :]), hidden)
            rnn_outputs.append(hidden.data)
        rnn_outputs = np.array(rnn_outputs)
        
        # Use only the last output from RNN for classification
        final_output = self.output_layer.forward(Tensor(rnn_outputs[-1]))
        return final_output
    
    def save(self, file_path):
        params = {
            'rnn': self.rnn.get_params(),
            'output': self.output_layer.get_params()
        }
        with open(file_path, 'wb') as f:
            pickle.dump(params, f)
    
    def load(self, file_path):
        with open(file_path, 'rb') as f:
            params = pickle.load(f)
        self.rnn.set_params(params['rnn'])
        self.output_layer.set_params(params['output'])

# Predefined dataset
def generate_predefined_dataset(num_samples, sequence_length, input_size, num_classes):
    X = np.random.randn(num_samples, sequence_length, input_size)
    y = np.random.randint(0, num_classes, size=(num_samples,))
    return X, y

# Parameters
input_size = 3
hidden_size = 4
output_size = 2
sequence_length = 5
num_samples = 100

# Generate predefined dataset
X, y = generate_predefined_dataset(num_samples, sequence_length, input_size, output_size)

# Create the network
network = RNNNetwork(input_size, hidden_size, output_size)

# Convert data to Tensor format
inputs = Tensor(X)
targets = Tensor(y)

# Forward pass
outputs = network.forward(inputs)
print("Output shape:", outputs.data.shape)
print("Sample output:\n", outputs.data[0])

# Calculate loss (using CrossEntropyLoss for demonstration)
loss_function = CrossEntropyLoss()
loss = loss_function.forward(outputs, targets)
print(f"Loss: {loss}")

# Save the model
network.save('rnn_model.pt')

# Create a new network and load the saved model
test_network = RNNNetwork(input_size, hidden_size, output_size)
test_network.load('rnn_model.pt')

# Generate test data
test_X, test_y = generate_predefined_dataset(10, sequence_length, input_size, output_size)
test_inputs = Tensor(test_X)

# Test the loaded network
test_outputs = test_network.forward(test_inputs)
print("\nTest output shape:", test_outputs.data.shape)
print("Sample test output:\n", test_outputs.data[0])