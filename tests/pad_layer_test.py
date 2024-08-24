from src.okrolearn.okrolearn import *

def test_padding_layers():
    # Create a small input tensor
    input_data = np.array([
        [[[1, 2, 3],
          [4, 5, 6],
          [7, 8, 9]]]
    ])
    input_tensor = Tensor(input_data)

    print("Input tensor:")
    print(input_tensor.data[0, 0])
    print()

    # Test ZeroPaddingLayer
    zero_pad = ZeroPaddingLayer(padding=1)
    zero_padded = zero_pad.forward(input_tensor)
    print("Zero padded output:")
    print(zero_padded.data[0, 0])
    print()

    # Test backward pass for ZeroPaddingLayer
    dL_dout = Tensor(np.ones_like(zero_padded.data))
    dL_dinput = zero_pad.backward(dL_dout)
    print("Zero padding gradient:")
    print(dL_dinput.data[0, 0])
    print()

    # Test ReflectionPaddingLayer
    reflect_pad = ReflectionPaddingLayer(padding=(1, 1, 2, 2))
    reflect_padded = reflect_pad.forward(input_tensor)
    print("Reflection padded output:")
    print(reflect_padded.data[0, 0])
    print()

    # Test backward pass for ReflectionPaddingLayer
    dL_dout = Tensor(np.ones_like(reflect_padded.data))
    dL_dinput = reflect_pad.backward(dL_dout)
    print("Reflection padding gradient:")
    print(dL_dinput.data[0, 0])
    print()

    # Test with non-square input
    non_square_input = np.array([
        [[[1, 2, 3, 4],
          [5, 6, 7, 8]]]
    ])
    non_square_tensor = Tensor(non_square_input)
    print("Non-square input tensor:")
    print(non_square_tensor.data[0, 0])
    print()

    # Test ZeroPaddingLayer with non-square input
    zero_pad_non_square = ZeroPaddingLayer(padding=(1, 2))
    zero_padded_non_square = zero_pad_non_square.forward(non_square_tensor)
    print("Zero padded non-square output:")
    print(zero_padded_non_square.data[0, 0])
    print()

# Run the test
if __name__ == "__main__":
    test_padding_layers()
