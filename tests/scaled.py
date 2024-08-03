from okrolearn.okrolearn import Tensor, ScaledDotProductAttention, np
import unittest


class TestScaledDotProductAttention(unittest.TestCase):

    def setUp(self):
        self.d_model = 4
        self.attention = ScaledDotProductAttention(self.d_model)

        # Initialize tensors for Q, K, V
        self.Q = Tensor(np.random.rand(2, self.d_model))
        self.K = Tensor(np.random.rand(2, self.d_model))
        self.V = Tensor(np.random.rand(2, self.d_model))

        # Mask is optional, initialize if needed
        self.mask = Tensor(np.zeros((2, 2)))

    def test_forward(self):
        output = self.attention.forward(self.Q, self.K, self.V, self.mask)

        self.assertIsInstance(output, Tensor)
        self.assertEqual(output.data.shape, (2, self.d_model))
        print("Forward output:", output.data)

    def test_backward(self):
        output = self.attention.forward(self.Q, self.K, self.V, self.mask)

        # Simulate gradient of loss with respect to the output
        dL_dout = Tensor(np.random.rand(2, self.d_model))
        lr = 0.01

        dL_dQ, dL_dK, dL_dV = self.attention.backward(dL_dout, lr)

        self.assertIsInstance(dL_dQ, Tensor)
        self.assertIsInstance(dL_dK, Tensor)
        self.assertIsInstance(dL_dV, Tensor)
        self.assertEqual(dL_dQ.data.shape, self.Q.data.shape)
        self.assertEqual(dL_dK.data.shape, self.K.data.shape)
        self.assertEqual(dL_dV.data.shape, self.V.data.shape)
        print("Backward dL_dQ:", dL_dQ.data)
        print("Backward dL_dK:", dL_dK.data)
        print("Backward dL_dV:", dL_dV.data)


if __name__ == '__main__':
    unittest.main()