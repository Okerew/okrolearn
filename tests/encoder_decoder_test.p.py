from okrolearn.okrolearn import Encoder, Decoder, Tensor, np
def test_encoder_decoder():
    input_dim = 784  # e.g., for MNIST images (28x28)
    hidden_dims = [256, 128]
    latent_dim = 64
    batch_size = 32

    encoder = Encoder(input_dim, hidden_dims, latent_dim)
    decoder = Decoder(latent_dim, hidden_dims[::-1], input_dim)

    # Create a random input
    x = Tensor(np.random.randn(batch_size, input_dim))

    # Forward pass through encoder
    encoded = encoder.forward(x)
    print(encoded.data.shape)
    assert encoded.data.shape == (batch_size, latent_dim), f"Encoded shape {encoded.data.shape} doesn't match expected shape {(batch_size, latent_dim)}"

    # Forward pass through decoder
    decoded = decoder.forward(encoded)
    print(decoded.data.shape)
    assert decoded.data.shape == (batch_size, input_dim), f"Decoded shape {decoded.data.shape} doesn't match expected shape {(batch_size, input_dim)}"

    print("Test passed successfully!")

# Run the test
test_encoder_decoder()