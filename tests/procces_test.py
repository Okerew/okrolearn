from okrolearn.okrolearn import Tensor
your_data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
target_data = [1, 2, 3]
# Preprocessing
X = Tensor(your_data)
X_preprocessed = X.preprocess(method='standardize')
print(X_preprocessed)

# Gradient Boosting
y = Tensor(target_data)
predictions, trees = X.gradient_boosting(y, n_estimators=100, learning_rate=0.1, max_depth=3)
print(predictions)
