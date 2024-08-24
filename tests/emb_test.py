from src.okrolearn.okrolearn import NeuralNetwork, Tensor, Embedding, np, MSELoss

def test_embedding_layer():
    # Test initialization
    vocab_size = 1000
    embedding_dim = 50
    embedding = Embedding(vocab_size, embedding_dim)
    assert embedding.embeddings.data.shape == (vocab_size, embedding_dim)

    # Test forward pass
    batch_size = 32
    sequence_length = 10
    input_indices = Tensor(np.random.randint(0, vocab_size, size=(batch_size, sequence_length)))
    output = embedding.forward(input_indices)
    assert output.data.shape == (batch_size, sequence_length, embedding_dim)

    # Test backward pass
    output_gradient = Tensor(np.random.randn(batch_size, sequence_length, embedding_dim))
    embedding.backward(output_gradient, lr=0.01)
    
    # Test integration with NeuralNetwork
    nn = NeuralNetwork()
    nn.add(embedding)
    loss_function = MSELoss()


    # Train the network
    inputs = Tensor(np.random.randint(0, vocab_size, size=(100, sequence_length)))
    targets = Tensor(np.random.randn(100, sequence_length, embedding_dim))
    nn.train(inputs, targets, epochs=5, lr=0.01, batch_size=32, loss_function=loss_function)

    print("All tests passed!")

# Run the test
test_embedding_layer()
