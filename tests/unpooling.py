from src.okrolearn.okrolearn import Tensor, np, MaxUnpoolingLayer, MaxPoolingLayer, AverageUnpoolingLayer, AveragePoolingLayer

def test_pooling_unpooling():
    # Create sample input data
    input_data = np.random.rand(1, 1, 6, 6)  # (batch_size, channels, height, width)
    input_tensor = Tensor(input_data)

    # Test Max Pooling and Unpooling
    max_pool = MaxPoolingLayer(pool_size=2, stride=2)
    max_unpool = MaxUnpoolingLayer(pool_size=2, stride=2)

    # Forward pass through max pooling
    max_pooled = max_pool.forward(input_tensor)
    print("Max Pooled shape:", max_pooled.data.shape)

    # Forward pass through max unpooling
    max_unpooled = max_unpool.forward(max_pooled)
    print("Max Unpooled shape:", max_unpooled.data.shape)

    # Test Average Pooling and Unpooling
    avg_pool = AveragePoolingLayer(pool_size=2, stride=2)
    avg_unpool = AverageUnpoolingLayer(pool_size=2, stride=2)

    # Forward pass through average pooling
    avg_pooled = avg_pool.forward(input_tensor)
    print("Average Pooled shape:", avg_pooled.data.shape)

    # Forward pass through average unpooling
    avg_unpooled = avg_unpool.forward(avg_pooled)
    print("Average Unpooled shape:", avg_unpooled.data.shape)

    # Verify shapes
    assert max_pooled.data.shape == (1, 1, 3, 3), "Max pooled shape is incorrect"
    assert max_unpooled.data.shape == (1, 1, 6, 6), "Max unpooled shape is incorrect"
    assert avg_pooled.data.shape == (1, 1, 3, 3), "Average pooled shape is incorrect"
    assert avg_unpooled.data.shape == (1, 1, 6, 6), "Average unpooled shape is incorrect"

    print("All shape tests passed!")

    # Optional: You can add more specific tests here, such as checking values

if __name__ == "__main__":
    test_pooling_unpooling()
