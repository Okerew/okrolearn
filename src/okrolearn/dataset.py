from okrolearn.okrolearn import Union, List, np, Tuple, Tensor


class Dataset:
    def __init__(self, data: Union[np.ndarray, List[np.ndarray]]):
        if isinstance(data, np.ndarray):
            self.data = [data]
        elif isinstance(data, list) and all(isinstance(item, np.ndarray) for item in data):
            self.data = data
        else:
            raise ValueError("Data must be a numpy array or a list of numpy arrays")

        self.tensors = [Tensor(arr) for arr in self.data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.tensors[idx]

    def to_tensors(self) -> List[Tensor]:
        return self.tensors

    def batch(self, batch_size: int) -> List[List[Tensor]]:
        batches = []
        for i in range(0, len(self), batch_size):
            batches.append(self.tensors[i:i + batch_size])
        return batches

    def shuffle(self):
        indices = np.random.permutation(len(self))
        self.data = [self.data[i] for i in indices]
        self.tensors = [self.tensors[i] for i in indices]

    def split(self, split_ratio: float) -> Tuple['Dataset', 'Dataset']:
        split_idx = int(len(self) * split_ratio)
        return Dataset(self.data[:split_idx]), Dataset(self.data[split_idx:])

    @classmethod
    def from_tensor_list(cls, tensor_list: List[Tensor]):
        return cls([tensor.data for tensor in tensor_list])

    @classmethod
    def from_numpy(cls, *arrays):
        return cls(list(arrays))

    def apply(self, func):
        self.data = [func(arr) for arr in self.data]
        self.tensors = [Tensor(arr) for arr in self.data]

    def __repr__(self):
        return f"Dataset(num_tensors={len(self)}, shapes={[arr.shape for arr in self.data]})"
