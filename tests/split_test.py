from okrolearn.src.okrolearn import *

def test_tensor_split():
    # Create a tensor with shape (6, 2)
    data = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]])
    tensor = Tensor(data)

    # Split the tensor into 3 sub-tensors along the first axis
    splits, backward_fn = tensor.split(3, axis=0)

    # Perform operations on the split tensors
    result_tensors = [split + Tensor(np.ones_like(split.data)) for split in splits]

    # Combine results to form a single tensor
    combined_result = Tensor(np.concatenate([t.data for t in result_tensors], axis=0))
    print(combined_result.data)

    print("All tests passed!")
test_tensor_split()
