from okrolearn.src.okrolearn.okrolearn import Tensor, np

def test_split():
    # Create a simple tensor
    data = np.array([[1, 2, 3, 4],
                     [5, 6, 7, 8],
                     [9, 10, 11, 12]])
    tensor = Tensor(data)

    # Test split into equal sections
    split_tensors = tensor.split(2, axis=1)
    
    assert len(split_tensors) == 2
    assert np.array_equal(split_tensors[0].data, np.array([[1, 2], [5, 6], [9, 10]]))
    assert np.array_equal(split_tensors[1].data, np.array([[3, 4], [7, 8], [11, 12]]))

    # Test backward pass for equal sections
    split_tensors[0].backward()
    split_tensors[1].backward()
    
    expected_grad = np.ones_like(data)
    assert np.array_equal(tensor.grad, expected_grad)
    # Reset grad for next test
    tensor.grad = None

    # Test split at specific indices
    split_tensors = tensor.split([1, 3], axis=1)
    
    assert len(split_tensors) == 3
    assert np.array_equal(split_tensors[0].data, np.array([[1], [5], [9]]))
    assert np.array_equal(split_tensors[1].data, np.array([[2, 3], [6, 7], [10, 11]]))
    assert np.array_equal(split_tensors[2].data, np.array([[4], [8], [12]]))

    # Test backward pass for specific indices
    split_tensors[0].backward()
    split_tensors[1].backward()
    split_tensors[2].backward()
    assert np.array_equal(tensor.grad, expected_grad)

    print("All tests passed!")

# Run the test
test_split()
