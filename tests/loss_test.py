import numpy as np
from okrolearn.src.okrolearn.okrolearn import *

def test_loss_functions():
    # Create some sample data
    outputs = Tensor(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
    targets = Tensor(np.array([[1.5, 2.5, 3.5], [3.5, 4.5, 5.5]]))

    # Test Mean Squared Logarithmic Error
    msle = MeanSquaredLogarithmicError()
    msle_loss = msle.forward(outputs, targets)
    msle_grad = msle.backward(outputs, targets)

    print("Mean Squared Logarithmic Error:")
    print(f"Loss: {msle_loss}")
    print(f"Gradient shape: {msle_grad.data.shape}")
    print(f"Gradient: {msle_grad.data}")
    print()

    # Test Mean Absolute Percentage Error
    mape = MeanAbsolutePercentageError()
    mape_loss = mape.forward(outputs, targets)
    mape_grad = mape.backward(outputs, targets)

    print("Mean Absolute Percentage Error:")
    print(f"Loss: {mape_loss}")
    print(f"Gradient shape: {mape_grad.data.shape}")
    print(f"Gradient: {mape_grad.data}")
    print()

    # Test Cosine Similarity Loss
    csl = CosineSimilarityLoss()
    csl_loss = csl.forward(outputs, targets)
    csl_grad = csl.backward(outputs, targets)

    print("Cosine Similarity Loss:")
    print(f"Loss: {csl_loss}")
    print(f"Gradient shape: {csl_grad.data.shape}")
    print(f"Gradient: {csl_grad.data}")
    print()

    mse = MSELoss()
    mse_loss = mse.forward(outputs, targets)
    mse_grad = mse.backward(outputs, targets)

    print("Mean Squared Error:")
    print(f"Loss: {mse_loss}")
    print(f"Gradient shape: {mse_grad.data.shape}")
    print(f"Gradient: {mse_grad.data}")
    print()

    # Test with edge cases
    edge_outputs = Tensor(np.array([[0.0, 1.0], [1.0, 0.0]]))
    edge_targets = Tensor(np.array([[0.0, 1.0], [1.0, 0.0]]))

    print("Edge case tests:")
    print(f"MSLE: {msle.forward(edge_outputs, edge_targets)}")
    print(f"MAPE: {mape.forward(edge_outputs, edge_targets)}")
    print(f"CSL: {csl.forward(edge_outputs, edge_targets)}")
    print(f"MSE: {mse.forward(edge_outputs, edge_targets)}")

if __name__ == "__main__":
    test_loss_functions()
