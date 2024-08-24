from src.okrolearn.okrolearn import Encoder, Decoder
from src.okrolearn.okroltext import TextToTensor, TensorToText
# Example usage:
vocabulary = ['a', 'b', 'c', ..., 'z', ' ', '<UNK>']  # Add all characters you expect in your text
text_to_tensor = TextToTensor(vocabulary)
tensor_to_text = TensorToText(vocabulary)

# Convert text to tensor
input_text = "hello world"
tensor = text_to_tensor.encode(input_text)

encoder = Encoder(input_dim=len(vocabulary), hidden_dims=[64, 32], output_dim=16)
decoder = Decoder(input_dim=16, hidden_dims=[32, 64], output_dim=len(vocabulary))

encoded = encoder.forward(tensor)
decoded = decoder.forward(encoded)

# Convert tensor back to text
output_text = tensor_to_text.decode(decoded)
print(output_text)
