from src.okrolearn.okrolearn import *


def test_transformer():
    print("Starting transformer test...")

    np.random.seed(42)

    input_dim = 512
    num_heads = 8
    ff_dim = 2048
    num_layers = 2
    output_dim = 1000
    batch_size = 32
    seq_length = 10

    print(
        f"Creating encoder and decoder with input_dim={input_dim}, num_heads={num_heads}, ff_dim={ff_dim}, num_layers={num_layers}")

    try:
        encoder = TransformerEncoder(input_dim, num_heads, ff_dim, num_layers)
        decoder = TransformerDecoder(input_dim, num_heads, ff_dim, num_layers, output_dim)
    except Exception as e:
        print(f"Error creating encoder or decoder: {e}")
        return

    print("Creating dummy input data...")
    encoder_input = Tensor(np.random.randn(batch_size, seq_length, input_dim))
    decoder_input = Tensor(np.random.randn(batch_size, seq_length, input_dim))

    print("Performing forward pass through encoder...")
    try:
        for i, layer in enumerate(encoder.layers):
            print(f"Processing encoder layer {i + 1}")
            attention_output = layer['attention'].forward(encoder_input, encoder_input, encoder_input)
            print(f"Attention output shape: {attention_output.data.shape}")

            norm1_output = layer['norm1'].forward(encoder_input + attention_output)
            print(f"Norm1 output shape: {norm1_output.data.shape}")

            ff_output = layer['ff'].forward(norm1_output)
            print(f"FF output shape: {ff_output.data.shape}")

            encoder_input = layer['norm2'].forward(norm1_output + ff_output)
            print(f"Norm2 output shape: {encoder_input.data.shape}")

        encoder_output = encoder_input
        print(f"Encoder output shape: {encoder_output.data.shape}")
    except Exception as e:
        print(f"Error in encoder forward pass: {e}")
        return

    print("Performing forward pass through decoder...")
    try:
        decoder_output = decoder.forward(decoder_input, encoder_output)
        print(f"Decoder output shape: {decoder_output.data.shape}")
    except Exception as e:
        print(f"Error in decoder forward pass: {e}")
        return

    print("All tests passed!")

    print("\nSample of decoder output:")
    print(decoder_output.data[0, 0, :10])


test_transformer()
