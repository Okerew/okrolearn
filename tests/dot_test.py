import unittest
from src.okrolearn.okrolearn import Tensor, ScaledDotProductAttention, np

class TestScaledDotProductAttention(unittest.TestCase):
    def setUp(self):
        self.d_model = 4
        self.attention = ScaledDotProductAttention(self.d_model)
        self.Q = Tensor(np.random.rand(2, self.d_model))
        self.K = Tensor(np.random.rand(2, self.d_model))
        self.V = Tensor(np.random.rand(2, self.d_model))
        self.mask = Tensor(np.array([[0, -1e9], [-1e9, 0]]), requires_grad=False)

    def test_forward(self):
        output = self.attention.forward(self.Q, self.K, self.V)
        print(output.data)
        self.assertEqual(output.data.shape, (2, self.d_model))

    def test_forward_with_mask(self):
        output = self.attention.forward(self.Q, self.K, self.V, self.mask)
        self.assertEqual(output.data.shape, (2, self.d_model))

    def test_backward(self):
        self.attention.forward(self.Q, self.K, self.V)
        dL_dout = Tensor(np.random.rand(2, self.d_model))
        print(dL_dout.data)

    def test_softmax(self):
        x = Tensor(np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]))
        softmax_result = self.attention.softmax(x)
        expected_result = np.array([[0.09003057, 0.24472847, 0.66524096],
                                    [0.09003057, 0.24472847, 0.66524096]])
        print(softmax_result.data)

if __name__ == '__main__':
    unittest.main()

