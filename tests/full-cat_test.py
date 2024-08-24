from src.okrolearn.okrolearn import Tensor, np
tensor1 = Tensor(np.random.rand(3, 4))
tensor2 = Tensor.full((3, 4), 1)
output = Tensor.cat([tensor1, tensor2], axis=1)
print(output)
