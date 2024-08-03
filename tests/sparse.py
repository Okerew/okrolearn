import unittest
from okrolearn.okrolearn import Tensor, np
from okrolearn.tensor import sparse

class TestTensorToSparse(unittest.TestCase):

    def setUp(self):
        # Create a sample sparse-like matrix for testing
        self.data = np.array([
            [1, 0, 0, 2],
            [0, 0, 3, 0],
            [4, 0, 0, 5]
        ])
        self.tensor = Tensor(self.data)

    def test_to_sparse_csr(self):
        sparse_matrix = self.tensor.to_sparse('csr')
        self.assertIsInstance(sparse_matrix, sparse.csr_matrix)
        np.testing.assert_array_equal(sparse_matrix.toarray(), self.data)

    def test_to_sparse_csc(self):
        sparse_matrix = self.tensor.to_sparse('csc')
        self.assertIsInstance(sparse_matrix, sparse.csc_matrix)
        np.testing.assert_array_equal(sparse_matrix.toarray(), self.data)

    def test_to_sparse_coo(self):
        sparse_matrix = self.tensor.to_sparse('coo')
        self.assertIsInstance(sparse_matrix, sparse.coo_matrix)
        np.testing.assert_array_equal(sparse_matrix.toarray(), self.data)

    def test_to_sparse_lil(self):
        sparse_matrix = self.tensor.to_sparse('lil')
        self.assertIsInstance(sparse_matrix, sparse.lil_matrix)
        np.testing.assert_array_equal(sparse_matrix.toarray(), self.data)

    def test_to_sparse_dok(self):
        sparse_matrix = self.tensor.to_sparse('dok')
        self.assertIsInstance(sparse_matrix, sparse.dok_matrix)
        np.testing.assert_array_equal(sparse_matrix.toarray(), self.data)

    def test_to_sparse_bsr(self):
        sparse_matrix = self.tensor.to_sparse('bsr')
        self.assertIsInstance(sparse_matrix, sparse.bsr_matrix)
        np.testing.assert_array_equal(sparse_matrix.toarray(), self.data)

    def test_to_sparse_default(self):
        sparse_matrix = self.tensor.to_sparse()
        self.assertIsInstance(sparse_matrix, sparse.csr_matrix)
        np.testing.assert_array_equal(sparse_matrix.toarray(), self.data)

    def test_to_sparse_invalid_format(self):
        with self.assertRaises(ValueError):
            self.tensor.to_sparse('invalid_format')

    def test_to_sparse_non_2d(self):
        tensor_1d = Tensor([1, 2, 3])
        with self.assertRaises(ValueError):
            tensor_1d.to_sparse()

class TestTensorFromSparse(unittest.TestCase):

    def setUp(self):
        self.data = np.array([
            [1, 0, 0, 2],
            [0, 0, 3, 0],
            [4, 0, 0, 5]
        ])
        self.sparse_matrix = sparse.csr_matrix(self.data)

    def test_from_sparse(self):
        tensor = Tensor.from_sparse(self.sparse_matrix)
        self.assertIsInstance(tensor, Tensor)
        np.testing.assert_array_equal(tensor.data, self.data)

    def test_from_sparse_invalid_input(self):
        with self.assertRaises(ValueError):
            Tensor.from_sparse(np.array([1, 2, 3]))  # Not a sparse matrix

    def test_roundtrip_sparse_conversion(self):
        original_tensor = Tensor(self.data)
        sparse_matrix = original_tensor.to_sparse()
        roundtrip_tensor = Tensor.from_sparse(sparse_matrix)
        np.testing.assert_array_equal(original_tensor.data, roundtrip_tensor.data)