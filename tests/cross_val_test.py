from okrolearn.okrolearn import Tensor, np

def mock_model(X_train, y_train):
    class MockModel:
        def __init__(self):
            pass

        def predict(self, X_test):
            # Mock prediction: just return zeros for demonstration
            return np.zeros(len(X_test))

    return MockModel()

# Generate sample data
X = np.random.rand(100, 5)
y = np.random.randint(0, 2, size=100)

# Create an instance of Tensor
tensor = Tensor(data=X)

# Call cross_val_score on the instance
scores = tensor.cross_val_score(mock_model, X, y, cv=5)
print("Cross-validation scores:", scores)
print("Mean score:", np.mean(scores))