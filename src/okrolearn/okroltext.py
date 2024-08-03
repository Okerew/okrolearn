from okrolearn.okrolearn import Tensor, List, np


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
        result = []
        for i in indices:
            index = int(i)
            if index not in self.index_to_char:
                print(f"Warning: Index {index} not found in index_to_char")
                char = '<UNK>'
            else:
                char = self.index_to_char[index]
            if not isinstance(char, str):
                print(f"Warning: Unexpected type for char: {type(char)}")
                char = str(char)
            result.append(char)
        return ''.join(result)
