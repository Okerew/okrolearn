from okrolearn.okrolearn import *

# Generate some dummy data
np.random.seed(0)
input_size = 20
output_size = 3
num_samples = 1000

X = np.random.randn(num_samples, input_size)
y = np.random.randint(0, output_size, size=num_samples)

assert X.shape[1] == input_size, f"Expected input size {input_size}, but got {X.shape[1]}"

# One-hot encode the targets for classification
y_one_hot = np.eye(output_size)[y]
print(f"Shape of y_one_hot: {y_one_hot.shape}")

# Convert one-hot encoded targets back to class indices
y_class_indices = np.argmax(y_one_hot, axis=1)

# Create and train the network
network = NeuralNetwork(temperature=0.8)
losses = network.train_with_suggested_architecture(
    inputs=X,
    targets=y_class_indices,  # Use class indices instead of one-hot encoded targets
    input_size=input_size,
    output_size=output_size,
    task_type='classification',
    data_type='tabular',
    depth=5,
    epochs=5,
    lr=0.2,
    batch_size=32,
)

print(f"Final loss: {losses[-1]}")
outputs = network.forward(Tensor(X))
print(outputs)
