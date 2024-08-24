from okrolearn.src.okrolearn.okrolearn import *
def test_channel_shuffle():
    # Create a Tensor with known values
    data = np.array([[[1, 2], [3, 4]],
                     [[5, 6], [7, 8]],
                     [[9, 10], [11, 12]],
                     [[13, 14], [15, 16]]])

    tensor = Tensor(data)
    
    # Define the number of groups for the shuffle
    groups = 2
    
    # Perform the channel shuffle
    shuffled_tensor = tensor.channel_shuffle(groups)
    print(shuffled_tensor.data)
    
    # Manually compute the expected result
    expected_data = np.array([[[1, 2], [3, 4]],
                              [[9, 10], [11, 12]],
                              [[5, 6], [7, 8]],
                              [[13, 14], [15, 16]]])
    
    # Verify the result
    assert np.array_equal(shuffled_tensor.data, expected_data), "Channel shuffle test failed!"
    print("Channel shuffle test passed!")
    
test_channel_shuffle()
