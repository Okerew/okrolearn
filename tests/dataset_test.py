from okrolearn.dataset import Dataset, np
# Create a dataset from numpy arrays
X = np.random.randn(100, 10)
y = np.random.randint(0, 2, (100, 1))
dataset = Dataset.from_numpy(X, y)

# Access individual tensors
X_tensor, y_tensor = dataset[0], dataset[1]

# Create batches
batches = dataset.batch(32)

# Shuffle the dataset
dataset.shuffle()

# Split the dataset
train_dataset, test_dataset = dataset.split(0.8)

# Apply a function to all arrays in the dataset
dataset.apply(lambda x: x * 2)

# Print dataset information
print(dataset)