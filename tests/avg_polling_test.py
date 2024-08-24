from okrolearn.src.okrolearn.okrolearn import Tensor, AveragePoolingLayer, np

def test_average_pooling():
    # Create a sample input (e.g., a 4x4 image with a single channel)
    sample_input = np.array([[[[1, 2, 3, 4],
                               [5, 6, 7, 8],
                               [9, 10, 11, 12],
                               [13, 14, 15, 16]]]])
    
    # Convert to Tensor
    inputs = Tensor(sample_input)
    
    # Initialize AveragePoolingLayer with pool size 2x2 and stride 2
    pool_size = 2
    stride = 2
    avg_pooling_layer = AveragePoolingLayer(pool_size=pool_size, stride=stride)
    
    # Forward pass through AveragePoolingLayer
    pooled_output = avg_pooling_layer.forward(inputs)
    
    # Print the shape and output of the pooling layer
    print("Input shape:", inputs.data.shape)
    print("Pooled output shape:", pooled_output.data.shape)
    print("Pooled output:\n", pooled_output.data)

if __name__ == "__main__":
    test_average_pooling()

