from okrolearn.src.okrolearn import Tensor, List, np


class TextToTensor:
    def __init__(self, vocabulary: List[str]):
        self.char_to_index = {char: i for i, char in enumerate(vocabulary)}
        self.index_to_char = {i: char for i, char in enumerate(vocabulary)}
        self.vocab_size = len(vocabulary)

    def encode(self, text: str) -> Tensor:
        indices = [self.char_to_index.get(char, self.char_to_index['<UNK>']) for char in text]
        one_hot = np.zeros((len(indices), self.vocab_size))
        for i, index in enumerate(indices):
            one_hot[i, index] = 1
        return Tensor(one_hot)


class TensorToText:
    def __init__(self, vocabulary: List[str]):
        self.char_to_index = {char: i for i, char in enumerate(vocabulary)}
        self.index_to_char = {i: char for i, char in enumerate(vocabulary)}

    def decode(self, tensor: Tensor) -> str:
        indices = tensor.data.argmax(axis=1)
        return ''.join(self.index_to_char[i] for i in indices)
