import unittest
from okrolearn.okrolearn import *

class TestInstanceNormLayer(unittest.TestCase):
    
    def setUp(self):
        self.num_features = 3
        self.layer = InstanceNormLayer(self.num_features)
        
        self.input_data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
        self.inputs = Tensor(self.input_data)  # assuming Tensor is properly defined
        
        # Mock gradients for backward pass
        self.dL_dout = Tensor(np.ones_like(self.input_data))
        self.lr = 0.1
    
    def test_forward(self):
        # Perform forward pass
        outputs = self.layer.forward(self.inputs)
        
        # Assertions
        self.assertEqual(outputs.data.shape, self.input_data.shape)  # Output shape should match input shape
    
    def test_backward(self):
        # Perform forward pass to initialize internal variables
        self.layer.forward(self.inputs)
        
        # Perform backward pass
        dL_dx = self.layer.backward(self.dL_dout, self.lr)
        
        # Assertions
        self.assertEqual(dL_dx.data.shape, self.input_data.shape)  # Gradient shape should match input shape
    
    def test_get_set_params(self):
        # Get initial parameters
        initial_params = self.layer.get_params()
        
        # Set new parameters (mocking a scenario where parameters are changed)
        new_params = {
            'gamma': np.ones((1, self.num_features)),
            'beta': np.zeros((1, self.num_features)),
        }
        self.layer.set_params(new_params)
        
        # Get parameters after setting
        updated_params = self.layer.get_params()
        
        # Assertions
        self.assertTrue(np.array_equal(updated_params['gamma'], new_params['gamma']))
        self.assertTrue(np.array_equal(updated_params['beta'], new_params['beta']))
    
if __name__ == '__main__':
    unittest.main()


