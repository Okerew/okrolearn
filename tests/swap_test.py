from src.okrolearn.okrolearn import *
x = Tensor(np.array([[1, 2], [3, 4]]))
y = x.swap(0, 1)  # This will transpose the matrix
print(y.data)
