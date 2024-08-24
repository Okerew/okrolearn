import random
import numpy as np
from typing import Tuple, Union, Callable, Optional, List
import matplotlib.pyplot as plt
import pstats
import io
import cProfile
import logging
import time
import pdb
from flask import Flask, request, jsonify
import pickle
from okrolearn.tensor import Tensor
import psutil


class DenseLayer:
    def __init__(self, input_size, output_size):
        """
        Paramaters:
        self input_size = the size of an input
        self output_size
        self weights = randomized input size and output_size with the numpy random algorithm * 0,1
        """
        self.weights = Tensor(np.random.randn(input_size, output_size) * 0.1)
        self.biases = Tensor(np.zeros((1, output_size)))

    def forward(self, inputs: Tensor):
        self.inputs = inputs
        self.outputs = inputs.dot(self.weights) + self.biases
        return self.outputs

    def backward(self, dL_dout: Tensor, lr: float):
        dL_dweights = self.inputs.transpose().dot(dL_dout)
        dL_dbiases = np.sum(dL_dout.data, axis=0, keepdims=True)
        dL_dinputs = dL_dout.dot(Tensor(self.weights.data.T))

        self.weights.grad = dL_dweights.data if self.weights.grad is None else self.weights.grad + dL_dweights.data
        self.biases.grad = dL_dbiases if self.biases.grad is None else self.biases.grad + dL_dbiases
        self.inputs.backward_fn = lambda grad: grad + dL_dinputs.data if self.inputs.backward_fn is None else lambda \
                x: self.inputs.backward_fn(x) + dL_dinputs.data

        self.weights.data -= lr * self.weights.grad
        self.biases.data -= lr * self.biases.grad

        return dL_dinputs

    def get_params(self):
        return {'weights': self.weights.data, 'biases': self.biases.data}

    def set_params(self, params):
        self.weights.data = params['weights']
        self.biases.data = params['biases']


class ELUActivationLayer:
    """
    Paramaters:
    self alpha = the alpha value
    self inputs = the inputs
    self outputs = the outputs
    """

    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def forward(self, inputs: Tensor):
        self.inputs = inputs
        self.outputs = inputs.apply(lambda x: x if x > 0 else self.alpha * (np.exp(
            x) - 1))  # If x is greater than 0, return x. If x is less than 0, return alpha * (exp(x) - 1)
        return self.outputs

    def backward(self, dL_dout: Tensor, lr: float):
        def elu_derivative(x):
            if x > 0:
                return 1
            else:
                return self.alpha * np.exp(x)

        dL_dinputs = dL_dout * self.inputs.apply(elu_derivative)
        self.inputs.grad = dL_dinputs.data if self.inputs.grad is None else self.inputs.grad + dL_dinputs.data  # Add the dL_dinputs to the grad
        self.inputs.backward_fn = lambda grad: grad + dL_dinputs.data if self.inputs.backward_fn is None else lambda \
                x: self.inputs.backward_fn(x) + dL_dinputs.data  # Add the dL_dinputs to the backward function
        return dL_dinputs

    def get_params(self):
        return None

    def set_params(self, params):
        pass


class ReLUActivationLayer:
    """
    Paramaters:
    self inputs = the inputs
    self outputs = the outputs
    """

    def forward(self, inputs: Tensor):
        self.inputs = inputs
        self.outputs = inputs.apply(lambda x: max(0, x))
        return self.outputs

    def backward(self, dL_dout: Tensor, lr: float):
        dL_dinputs = dL_dout * self.inputs.apply(
            lambda x: 1 if x > 0 else 0)  # If x is greater than 0, return 1. If x is less than 0, return 0

        self.inputs.grad = dL_dinputs.data if self.inputs.grad is None else self.inputs.grad + dL_dinputs.data
        self.inputs.backward_fn = lambda grad: grad + dL_dinputs.data if self.inputs.backward_fn is None else lambda \
                x: self.inputs.backward_fn(x) + dL_dinputs.data  # Add the dL_dinputs to the backward function

        return dL_dinputs

    def get_params(self):
        return None

    def set_params(self, params):
        pass


class SwishActivationLayer:
    """
    Parameters:
    self.inputs = the inputs
    self.outputs = the outputs
    """

    def sigmoid(self, x):
        # Clip values to avoid overflow
        return 1 / (1 + np.exp(-np.clip(x, -88, 88)))

    def forward(self, inputs: Tensor):
        self.inputs = inputs
        self.sigmoid_values = inputs.apply(self.sigmoid)
        self.outputs = inputs * self.sigmoid_values
        return self.outputs

    def backward(self, dL_dout: Tensor, lr: float):
        swish = self.outputs.data
        sigmoid = self.sigmoid_values.data

        # Compute dswish/dx with overflow protection
        dswish_dx = swish + sigmoid * (1 - swish)

        # Clip the result to avoid extreme values
        dswish_dx = np.clip(dswish_dx, -1e9, 1e9)

        dL_dinputs = Tensor(dL_dout.data * dswish_dx)

        self.inputs.grad = dL_dinputs.data if self.inputs.grad is None else self.inputs.grad + dL_dinputs.data
        self.inputs.backward_fn = lambda grad: grad + dL_dinputs.data if self.inputs.backward_fn is None else lambda \
                x: self.inputs.backward_fn(x) + dL_dinputs.data

        return dL_dinputs

    def get_params(self):
        return None

    def set_params(self, params):
        pass


class SELUActivationLayer:
    """
    Parameters:
    self.inputs = the inputs
    self.outputs = the outputs
    """

    def __init__(self, alpha=1.6732632423543772848170429916717, scale=1.0507009873554804934193349852946):
        self.alpha = alpha
        self.scale = scale

    def forward(self, inputs: Tensor):
        self.inputs = inputs
        self.outputs = inputs.apply(lambda x: self.scale * (x if x > 0 else self.alpha * (np.exp(x) - 1)))
        return self.outputs

    def backward(self, dL_dout: Tensor, lr: float):
        dL_dinputs = dL_dout * self.inputs.apply(
            lambda x: self.scale if x > 0 else self.scale * self.alpha * np.exp(x))

        self.inputs.grad = dL_dinputs.data if self.inputs.grad is None else self.inputs.grad + dL_dinputs.data
        self.inputs.backward_fn = lambda grad: grad + dL_dinputs.data if self.inputs.backward_fn is None else lambda \
                x: self.inputs.backward_fn(x) + dL_dinputs.data

        return dL_dinputs

    def get_params(self):
        return None

    def set_params(self, params):
        pass


class LeakyReLUActivationLayer:
    """
    Paramaters:
    self inputs = the inputs
    self outputs = the outputs
    self alpha = the alpha value

    It is simmilar to ReLUActivationLayer
    """

    def __init__(self, alpha: float = 0.01):
        self.alpha = alpha

    def forward(self, inputs: Tensor):
        self.inputs = inputs
        self.outputs = inputs.apply(lambda x: x if x > 0 else self.alpha * x)
        return self.outputs

    def backward(self, dL_dout: Tensor, lr: float):
        dL_dinputs = dL_dout * self.inputs.apply(lambda x: 1 if x > 0 else self.alpha)

        self.inputs.grad = dL_dinputs.data if self.inputs.grad is None else self.inputs.grad + dL_dinputs.data
        self.inputs.backward_fn = lambda grad: grad + dL_dinputs.data if self.inputs.backward_fn is None else lambda \
                x: self.inputs.backward_fn(x) + dL_dinputs.data

        return dL_dinputs

    def get_params(self):
        return None

    def set_params(self, params):
        pass


class PReLUActivationLayer:
    """
    Paramaters:
    self inputs = the inputs
    self outputs = the outputs
    self alpha = the alpha value
    """

    def __init__(self, alpha: float = 0.01):
        self.alpha = alpha
        self.d_alpha = 0

    def forward(self, inputs: Tensor):
        self.inputs = inputs
        self.outputs = inputs.apply(lambda x: x if x > 0 else self.alpha * x)
        return self.outputs

    def backward(self, dL_dout: Tensor, lr: float):
        dL_dinputs = dL_dout * self.inputs.apply(lambda x: 1 if x > 0 else self.alpha)

        grad_alpha = dL_dout * self.inputs.apply(lambda x: x if x <= 0 else 0)
        self.d_alpha = np.sum(grad_alpha.data)
        self.alpha -= lr * self.d_alpha

        if self.inputs.grad is None:
            self.inputs.grad = dL_dinputs.data
        else:
            self.inputs.grad += dL_dinputs.data

        if self.inputs.backward_fn is None:
            self.inputs.backward_fn = lambda grad: dL_dinputs.data  # Add the dL_dinputs to the backward function
        else:
            prev_backward_fn = self.inputs.backward_fn
            self.inputs.backward_fn = lambda grad: prev_backward_fn(
                grad) + dL_dinputs.data  # Add the dL_dinputs to the backward function

        return dL_dinputs

    def get_params(self):
        return {'alpha': self.alpha}

    def set_params(self, params):
        if params and 'alpha' in params:
            self.alpha = params['alpha']


class GELUActivationLayer:
    """
    GELU (Gaussian Error Linear Unit) Activation Layer.

    Parameters:
    self.inputs: The inputs to the layer.
    self.outputs: The outputs from the layer.
    """

    def __init__(self):
        self.sqrt_2_over_pi = np.sqrt(2 / np.pi)  # Precompute constant
        self.tanh_clip_value = 15  # Clipping value for tanh argument to prevent overflow

    def forward(self, inputs):
        self.inputs = inputs
        x = inputs.data

        # GELU activation function: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        x_cubed = np.power(x, 3)
        tanh_arg = self.sqrt_2_over_pi * (x + 0.044715 * x_cubed)

        # Clip the tanh argument to avoid overflow issues
        tanh_arg = np.clip(tanh_arg, -self.tanh_clip_value, self.tanh_clip_value)

        self.outputs = Tensor(0.5 * x * (1 + np.tanh(tanh_arg)), requires_grad=inputs.requires_grad)
        return self.outputs

    def backward(self, dL_dout, lr):
        x = self.inputs.data
        dL_dout = dL_dout.data

        # GELU derivative
        x_cubed = np.power(x, 3)
        tanh_arg = self.sqrt_2_over_pi * (x + 0.044715 * x_cubed)

        # Clip the tanh argument to avoid overflow issues
        tanh_arg = np.clip(tanh_arg, -self.tanh_clip_value, self.tanh_clip_value)
        tanh_part = np.tanh(tanh_arg)

        sech2_tanh = 1 - tanh_part ** 2
        gelu_derivative = 0.5 * tanh_part + 0.5 * x * sech2_tanh * self.sqrt_2_over_pi * (
                    1 + 3 * 0.044715 * np.power(x, 2))

        dL_dinputs = dL_dout * gelu_derivative

        if self.inputs.requires_grad:
            self.inputs.grad = dL_dinputs

        return Tensor(dL_dinputs)

    def get_params(self):
        return None

    def set_params(self, params):
        pass


class SoftsignActivationLayer:
    def forward(self, inputs: Tensor):
        self.inputs = inputs
        self.outputs = Tensor(inputs.data / (1 + np.abs(inputs.data)))
        return self.outputs

    def backward(self, dL_dout: Tensor, lr: float = None):
        gradients = dL_dout.data * self._compute_derivative()
        return Tensor(gradients)

    def _compute_derivative(self):
        return 1 / ((1 + np.abs(self.inputs.data)) ** 2)

    def get_params(self):
        return None

    def set_params(self, params: dict):
        pass


class SoftmaxActivationLayer:
    """
    Softmax activation layer for neural networks.

    Parameters:
    self.inputs: Tensor - the inputs
    self.outputs: Tensor - the outputs after applying softmax
    """

    def forward(self, inputs: Tensor):
        # Subtract max value for numerical stability
        shifted_inputs = inputs.data - np.max(inputs.data, axis=1, keepdims=True)
        exp_values = np.exp(shifted_inputs)
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)

        # Ensure no NaNs or Infs in outputs
        probabilities = np.nan_to_num(probabilities, nan=0.0, posinf=1.0, neginf=0.0)

        self.outputs = Tensor(probabilities)
        return self.outputs

    def backward(self, dL_dout: Tensor, lr: float = None):
        jacobians = self._compute_jacobians()
        gradients = np.einsum('ijk,ik->ij', jacobians, dL_dout.data)
        return Tensor(gradients)

    def _compute_jacobians(self):
        batch_size, num_classes = self.outputs.data.shape
        jacobians = np.zeros((batch_size, num_classes, num_classes))

        for i in range(batch_size):
            for j in range(num_classes):
                for k in range(num_classes):
                    if j == k:
                        jacobians[i, j, k] = self.outputs.data[i, j] * (1 - self.outputs.data[i, k])
                    else:
                        jacobians[i, j, k] = -self.outputs.data[i, j] * self.outputs.data[i, k]

        return jacobians

    def get_params(self):
        return None

    def set_params(self, params: dict):
        pass


class SoftminActivationLayer:
    """
    Softmin activation layer for neural networks.

    Parameters:
    self.inputs: Tensor - the inputs
    self.outputs: Tensor - the outputs after applying softmin
    """

    def forward(self, inputs: Tensor):
        # Add max value for numerical stability (opposite of softmax)
        shifted_inputs = inputs.data + np.max(inputs.data, axis=1, keepdims=True)
        inv_exp_values = np.exp(-shifted_inputs)
        probabilities = inv_exp_values / np.sum(inv_exp_values, axis=1, keepdims=True)

        # Ensure no NaNs or Infs in outputs
        probabilities = np.nan_to_num(probabilities, nan=0.0, posinf=1.0, neginf=0.0)

        self.outputs = Tensor(probabilities)
        return self.outputs

    def backward(self, dL_dout: Tensor, lr: float = None):
        jacobians = self._compute_jacobians()
        gradients = np.einsum('ijk,ik->ij', jacobians, dL_dout.data)
        return Tensor(gradients)

    def _compute_jacobians(self):
        batch_size, num_classes = self.outputs.data.shape
        jacobians = np.zeros((batch_size, num_classes, num_classes))

        for i in range(batch_size):
            for j in range(num_classes):
                for k in range(num_classes):
                    if j == k:
                        jacobians[i, j, k] = self.outputs.data[i, j] * (1 - self.outputs.data[i, k])
                    else:
                        jacobians[i, j, k] = -self.outputs.data[i, j] * self.outputs.data[i, k]

        return jacobians

    def get_params(self):
        return None

    def set_params(self, params: dict):
        pass


class LogSoftmaxActivationLayer:
    """
    Parameters:
    self.inputs = the inputs
    self.outputs = the outputs
    """

    def forward(self, inputs: Tensor):
        # Convert data to float64 to avoid large integers
        inputs.data = inputs.data.astype(np.float64)

        # Subtract max for numerical stability
        max_vals = np.max(inputs.data, axis=1, keepdims=True)

        # Compute exp values and ensure they are numerically stable
        exp_values = np.exp(np.clip(inputs.data - max_vals, -500, 500))

        # Sum exp values
        sum_exp_values = np.sum(exp_values, axis=1, keepdims=True)

        # Compute log softmax
        log_softmax = (inputs.data - max_vals) - np.log(sum_exp_values)
        self.outputs = Tensor(log_softmax)
        return self.outputs

    def backward(self, dL_dout: Tensor, lr: float = None):
        jacobians = self._compute_jacobians()
        gradients = np.einsum('ijk,ik->ij', jacobians, dL_dout.data)
        return Tensor(gradients)

    def _compute_jacobians(self):
        batch_size, num_classes = self.outputs.data.shape
        jacobians = np.zeros((batch_size, num_classes, num_classes), dtype=np.float64)

        for i in range(batch_size):
            for j in range(num_classes):
                for k in range(num_classes):
                    if j == k:
                        jacobians[i, j, k] = 1 - np.exp(self.outputs.data[i, j])
                    else:
                        jacobians[i, j, k] = -np.exp(self.outputs.data[i, k])

        return jacobians

    def get_params(self):
        return None

    def set_params(self, params: dict):
        pass


class SigmoidActivationLayer:
    """
    Parameters:
    self.inputs = the inputs
    self.outputs = the outputs
    """

    def forward(self, inputs: Tensor):
        self.inputs = inputs
        sigmoid_values = 1 / (1 + np.exp(-inputs.data))
        self.outputs = Tensor(sigmoid_values)
        return self.outputs

    def backward(self, dL_dout: Tensor, lr: float = None):
        sigmoid_derivative = self.outputs.data * (1 - self.outputs.data)
        gradients = dL_dout.data * sigmoid_derivative
        return Tensor(gradients)

    def get_params(self):
        return None

    def set_params(self, params: dict):
        pass


class Fold:
    def __init__(self, output_size, kernel_size, stride=1, padding=0):
        self.output_size = output_size
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.input_shape = None

    def forward(self, inputs: Tensor):
        self.input_shape = inputs.data.shape
        batch_size, n_channels, length = inputs.data.shape

        # Calculate output dimensions
        height, width = self.output_size
        out_height = (height + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_width = (width + 2 * self.padding - self.kernel_size) // self.stride + 1

        # Reshape the input
        folded = inputs.data.reshape(batch_size, n_channels // (self.kernel_size ** 2),
                                     self.kernel_size, self.kernel_size, out_height, out_width)

        # Transpose and reshape to get the final output
        folded = folded.transpose(0, 1, 4, 2, 5, 3).reshape(batch_size, -1, height, width)

        return Tensor(folded)

    def backward(self, dL_dout: Tensor):
        batch_size, _, height, width = dL_dout.data.shape

        # Calculate output dimensions
        out_height = (height + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_width = (width + 2 * self.padding - self.kernel_size) // self.stride + 1

        # Reshape to match the unfolded shape
        grad_reshaped = dL_dout.data.reshape(batch_size, -1, out_height, self.kernel_size, out_width, self.kernel_size)
        grad_transposed = grad_reshaped.transpose(0, 1, 3, 5, 2, 4)

        # Reshape to match the input shape
        unfolded = grad_transposed.reshape(self.input_shape)

        return Tensor(unfolded)


class Unfold:
    def __init__(self, kernel_size, stride=1, padding=0):
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.input_shape = None

    def forward(self, inputs: Tensor):
        self.input_shape = inputs.data.shape
        batch_size, n_channels, height, width = inputs.data.shape

        # Calculate output dimensions
        out_height = (height + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_width = (width + 2 * self.padding - self.kernel_size) // self.stride + 1

        # Perform the unfolding operation
        unfolded = np.zeros((batch_size, n_channels * self.kernel_size * self.kernel_size,
                             out_height * out_width))

        for i in range(out_height):
            for j in range(out_width):
                h_start = i * self.stride - self.padding
                w_start = j * self.stride - self.padding
                h_end = h_start + self.kernel_size
                w_end = w_start + self.kernel_size

                # Extract the patch
                patch = inputs.data[:, :, max(0, h_start):min(height, h_end),
                        max(0, w_start):min(width, w_end)]

                # Pad the patch if necessary
                if patch.shape[2] < self.kernel_size or patch.shape[3] < self.kernel_size:
                    padded_patch = np.zeros((batch_size, n_channels, self.kernel_size, self.kernel_size))
                    padded_patch[:, :, :patch.shape[2], :patch.shape[3]] = patch
                    patch = padded_patch

                # Flatten the patch and store it in the output
                unfolded[:, :, i * out_width + j] = patch.reshape(batch_size, -1)

        return Tensor(unfolded)

    def backward(self, dL_dout: Tensor):
        batch_size, _, _ = dL_dout.data.shape
        _, n_channels, height, width = self.input_shape

        # Initialize the gradient with respect to input
        dx = np.zeros(self.input_shape)

        # Calculate output dimensions
        out_height = (height + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_width = (width + 2 * self.padding - self.kernel_size) // self.stride + 1

        # Reshape dL_dout
        dL_dout = dL_dout.data.reshape(batch_size, n_channels, self.kernel_size, self.kernel_size,
                                       out_height, out_width)

        # Perform the folding operation
        for i in range(out_height):
            for j in range(out_width):
                h_start = i * self.stride - self.padding
                w_start = j * self.stride - self.padding
                h_end = h_start + self.kernel_size
                w_end = w_start + self.kernel_size

                # Add the gradient to the appropriate location
                dx[:, :, max(0, h_start):min(height, h_end),
                max(0, w_start):min(width, w_end)] += dL_dout[:, :, :, :, i, j]

        return Tensor(dx)


class FlattenLayer:
    def __init__(self):
        self.inputs_shape = None

    def forward(self, inputs: Tensor):
        self.inputs_shape = inputs.data.shape
        batch_size = inputs.data.shape[0]
        flattened_shape = (batch_size,
                           -1)  # Reshape shape by -1 leaving the batch_size dimension intact and flattening the remaining dimensions
        return inputs.reshape(flattened_shape)

    def backward(self, dL_dout: Tensor, lr: float):
        return dL_dout.reshape(self.inputs_shape)


class UnflattenLayer:
    def __init__(self, output_shape):
        self.output_shape = output_shape

    def forward(self, inputs: Tensor):
        batch_size = inputs.data.shape[0]
        unflattened_shape = (batch_size,) + self.output_shape
        return inputs.reshape(unflattened_shape)

    def backward(self, dL_dout: Tensor, lr: float):
        batch_size = dL_dout.data.shape[0]
        flattened_shape = (batch_size, -1)  # Doing the opposite of flattening
        return dL_dout.reshape(flattened_shape)


class Conv1DLayer:
    """
    Parameters:
    self weights = the weights
    self biases = the biases
    self in channels = input channels
    self out channels = output channels
    self padding = the padding
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.weights = Tensor(np.random.randn(out_channels, in_channels, kernel_size).astype(
            np.float32) * 0.1)  # weights is equal to np.random.randn(out_channels, in_channels, kernel_size).astype(np.float32) * 0.1
        self.biases = Tensor(np.zeros((out_channels, 1),
                                      dtype=np.float32))  # biases is equal to np.zeros((out_channels, 1), dtype=np.float32)

    def forward(self, inputs: Tensor):
        self.inputs = inputs
        batch_size, in_channels, input_length = inputs.data.shape

        output_length = ((input_length + 2 * self.padding - self.kernel_size) // self.stride) + 1

        if self.padding > 0:
            padded_inputs = np.pad(inputs.data, ((0, 0), (0, 0), (self.padding, self.padding)), mode='constant')
        else:
            padded_inputs = inputs.data

        output = np.zeros((batch_size, self.out_channels, output_length), dtype=np.float32)
        for i in range(output_length):
            start = i * self.stride
            end = start + self.kernel_size
            output[:, :, i] = np.sum(
                padded_inputs[:, np.newaxis, :, start:end] * self.weights.data[np.newaxis, :, :, :],
                axis=(2, 3)
            )
            # Sums the product of the padded inputs and the weights along an axis

        output += self.biases.data.reshape(1, -1, 1)

        self.output = Tensor(output)
        return self.output

    def backward(self, dL_dout: Tensor, lr: float):
        batch_size, _, output_length = dL_dout.data.shape

        dL_dweights = np.zeros_like(self.weights.data)
        dL_dbiases = np.sum(dL_dout.data, axis=(0, 2), keepdims=True)
        dL_dinputs = np.zeros_like(self.inputs.data)

        if self.padding > 0:
            padded_inputs = np.pad(self.inputs.data, ((0, 0), (0, 0), (self.padding, self.padding)), mode='constant')
        else:
            padded_inputs = self.inputs.data

        for i in range(output_length):
            start = i * self.stride
            end = start + self.kernel_size
            dL_dweights += np.sum(
                padded_inputs[:, np.newaxis, :, start:end] * dL_dout.data[:, :, i:i + 1, np.newaxis],
                axis=0
            )
            dL_dinputs[:, :, start:end] += np.sum(
                self.weights.data[np.newaxis, :, :, :] * dL_dout.data[:, :, i:i + 1, np.newaxis],
                axis=1
            )

        if self.padding > 0:
            dL_dinputs = dL_dinputs[:, :, self.padding:-self.padding]

        self.weights.grad = dL_dweights if self.weights.grad is None else self.weights.grad + dL_dweights
        self.biases.grad = dL_dbiases if self.biases.grad is None else self.biases.grad + dL_dbiases

        self.weights.data -= lr * self.weights.grad
        self.biases.data -= lr * self.biases.grad

        self.inputs.grad = dL_dinputs if self.inputs.grad is None else self.inputs.grad + dL_dinputs
        self.inputs.backward_fn = lambda grad: grad + dL_dinputs if self.inputs.backward_fn is None else lambda \
                x: self.inputs.backward_fn(x) + dL_dinputs

        return Tensor(dL_dinputs)

    def get_params(self):
        return {
            'weights': self.weights.data,
            'biases': self.biases.data
        }

    def set_params(self, params):
        self.weights.data = params['weights']
        self.biases.data = params['biases']


class Conv2DLayer:
    # The same as Conv1d but multiplied along 2 dimensions
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, transposed=False):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.transposed = transposed

        if transposed:
            self.filters = Tensor(np.random.randn(in_channels, out_channels, *self.kernel_size) * 0.1)
        else:
            self.filters = Tensor(np.random.randn(out_channels, in_channels, *self.kernel_size) * 0.1)
        self.biases = Tensor(np.zeros((out_channels, 1)))

    def forward(self, inputs: Tensor):
        self.inputs = inputs
        if self.transposed:
            return self.forward_transposed(inputs)
        else:
            return self.forward_normal(inputs)

    def forward_normal(self, inputs: Tensor):
        batch_size, in_channels, in_height, in_width = inputs.data.shape
        out_height = (in_height + 2 * self.padding[0] - self.kernel_size[0]) // self.stride[0] + 1
        out_width = (in_width + 2 * self.padding[1] - self.kernel_size[1]) // self.stride[1] + 1

        padded_inputs = np.pad(inputs.data,
                               ((0, 0), (0, 0), (self.padding[0], self.padding[0]), (self.padding[1], self.padding[1])),
                               mode='constant')

        output = np.zeros((batch_size, self.out_channels, out_height, out_width))

        for i in range(out_height):
            for j in range(out_width):
                h_start = i * self.stride[0]
                h_end = h_start + self.kernel_size[0]
                w_start = j * self.stride[1]
                w_end = w_start + self.kernel_size[1]

                receptive_field = padded_inputs[:, :, h_start:h_end, w_start:w_end]
                output[:, :, i, j] = np.sum(receptive_field[:, np.newaxis, :, :, :] * self.filters.data, axis=(2, 3, 4))

        output += self.biases.data.reshape(1, -1, 1, 1)
        self.outputs = Tensor(output)
        return self.outputs

    def forward_transposed(self, inputs: Tensor):
        batch_size, in_channels, in_height, in_width = inputs.data.shape
        out_height = (in_height - 1) * self.stride[0] - 2 * self.padding[0] + self.kernel_size[0]
        out_width = (in_width - 1) * self.stride[1] - 2 * self.padding[1] + self.kernel_size[1]

        output = np.zeros((batch_size, self.out_channels, out_height, out_width))

        for i in range(in_height):
            for j in range(in_width):
                h_start = i * self.stride[0]
                h_end = h_start + self.kernel_size[0]
                w_start = j * self.stride[1]
                w_end = w_start + self.kernel_size[1]

                output[:, :, h_start:h_end, w_start:w_end] += np.sum(
                    inputs.data[:, :, i, j][:, :, np.newaxis, np.newaxis, np.newaxis] * self.filters.data,
                    axis=1
                )

        if self.padding[0] > 0 or self.padding[1] > 0:
            output = output[:, :, self.padding[0]:output.shape[2] - self.padding[0],
                     self.padding[1]:output.shape[3] - self.padding[1]]

        output += self.biases.data.reshape(1, -1, 1, 1)
        self.outputs = Tensor(output)
        return self.outputs

    def backward(self, dL_dout: Tensor, lr: float):
        if self.transposed:
            return self.backward_transposed(dL_dout, lr)
        else:
            return self.backward_normal(dL_dout, lr)

    def backward_normal(self, dL_dout: Tensor, lr: float):
        batch_size, _, out_height, out_width = dL_dout.data.shape
        padded_inputs = np.pad(self.inputs.data,
                               ((0, 0), (0, 0), (self.padding[0], self.padding[0]), (self.padding[1], self.padding[1])),
                               mode='constant')

        dL_dfilters = np.zeros_like(self.filters.data)
        dL_dinputs = np.zeros_like(padded_inputs)

        for i in range(out_height):
            for j in range(out_width):
                h_start = i * self.stride[0]
                h_end = h_start + self.kernel_size[0]
                w_start = j * self.stride[1]
                w_end = w_start + self.kernel_size[1]

                receptive_field = padded_inputs[:, :, h_start:h_end, w_start:w_end]
                dL_dfilters += np.sum(receptive_field[:, np.newaxis, :, :, :] * dL_dout.data[:, :, i:i + 1, j:j + 1],
                                      axis=0)
                dL_dinputs[:, :, h_start:h_end, w_start:w_end] += np.sum(
                    self.filters.data[np.newaxis, :, :, :, :] * dL_dout.data[:, :, i:i + 1, j:j + 1], axis=1)

        dL_dbiases = np.sum(dL_dout.data, axis=(0, 2, 3), keepdims=True)

        self.filters.grad = dL_dfilters if self.filters.grad is None else self.filters.grad + dL_dfilters
        self.biases.grad = dL_dbiases if self.biases.grad is None else self.biases.grad + dL_dbiases
        self.inputs.backward_fn = lambda grad: grad + dL_dinputs if self.inputs.backward_fn is None else lambda \
                x: self.inputs.backward_fn(x) + dL_dinputs

        self.filters.data -= lr * self.filters.grad
        self.biases.data -= lr * self.biases.grad

        if self.padding[0] > 0 or self.padding[1] > 0:
            dL_dinputs = dL_dinputs[:, :, self.padding[0]:-self.padding[0], self.padding[1]:-self.padding[1]]

        return Tensor(dL_dinputs)

    def backward_transposed(self, dL_dout: Tensor, lr: float):
        batch_size, _, out_height, out_width = dL_dout.data.shape
        in_height = (out_height + 2 * self.padding[0] - self.kernel_size[0]) // self.stride[0] + 1
        in_width = (out_width + 2 * self.padding[1] - self.kernel_size[1]) // self.stride[1] + 1

        dL_dfilters = np.zeros_like(self.filters.data)
        dL_dinputs = np.zeros((batch_size, self.in_channels, in_height, in_width))

        padded_dL_dout = np.pad(dL_dout.data, (
            (0, 0), (0, 0), (self.padding[0], self.padding[0]), (self.padding[1], self.padding[1])), mode='constant')

        for i in range(in_height):
            for j in range(in_width):
                h_start = i * self.stride[0]
                h_end = h_start + self.kernel_size[0]
                w_start = j * self.stride[1]
                w_end = w_start + self.kernel_size[1]

                dL_dfilters += np.sum(
                    self.inputs.data[:, :, i:i + 1, j:j + 1][:, :, np.newaxis, np.newaxis, np.newaxis] * padded_dL_dout[
                                                                                                         :, np.newaxis,
                                                                                                         :,
                                                                                                         h_start:h_end,
                                                                                                         w_start:w_end],
                    axis=0
                )
                dL_dinputs[:, :, i, j] = np.sum(
                    self.filters.data * padded_dL_dout[:, np.newaxis, :, h_start:h_end, w_start:w_end],
                    axis=(2, 3, 4)
                )

        dL_dbiases = np.sum(dL_dout.data, axis=(0, 2, 3), keepdims=True)

        self.filters.grad = dL_dfilters if self.filters.grad is None else self.filters.grad + dL_dfilters
        self.biases.grad = dL_dbiases if self.biases.grad is None else self.biases.grad + dL_dbiases
        self.inputs.backward_fn = lambda grad: grad + dL_dinputs if self.inputs.backward_fn is None else lambda \
                x: self.inputs.backward_fn(x) + dL_dinputs

        self.filters.data -= lr * self.filters.grad
        self.biases.data -= lr * self.biases.grad

        return Tensor(dL_dinputs)

    def get_params(self):
        return {'filters': self.filters.data, 'biases': self.biases.data}

    def set_params(self, params):
        self.filters.data = params['filters']
        self.biases.data = params['biases']


class Conv3DLayer:
    # The same as conv1d but with *3 dimensions
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding, padding)

        self.filters = Tensor(np.random.randn(out_channels, in_channels, *self.kernel_size).astype(np.float32) * 0.1)
        self.biases = Tensor(np.zeros((out_channels, 1), dtype=np.float32))

    def forward(self, inputs: Tensor):
        self.inputs = inputs
        batch_size, in_channels, in_depth, in_height, in_width = inputs.data.shape

        out_depth = ((in_depth + 2 * self.padding[0] - self.kernel_size[0]) // self.stride[0]) + 1
        out_height = ((in_height + 2 * self.padding[1] - self.kernel_size[1]) // self.stride[1]) + 1
        out_width = ((in_width + 2 * self.padding[2] - self.kernel_size[2]) // self.stride[2]) + 1

        if any(p > 0 for p in self.padding):
            padded_inputs = np.pad(inputs.data, ((0, 0), (0, 0),
                                                 (self.padding[0], self.padding[0]),
                                                 (self.padding[1], self.padding[1]),
                                                 (self.padding[2], self.padding[2])), mode='constant')
        else:
            padded_inputs = inputs.data

        output = np.zeros((batch_size, self.out_channels, out_depth, out_height, out_width), dtype=np.float32)

        for i in range(out_depth):
            for j in range(out_height):
                for k in range(out_width):
                    d_start = i * self.stride[0]
                    d_end = d_start + self.kernel_size[0]
                    h_start = j * self.stride[1]
                    h_end = h_start + self.kernel_size[1]
                    w_start = k * self.stride[2]
                    w_end = w_start + self.kernel_size[2]

                    output[:, :, i, j, k] = np.sum(
                        padded_inputs[:, np.newaxis, :, d_start:d_end, h_start:h_end, w_start:w_end] *
                        self.filters.data[np.newaxis, :, :, :, :, :],
                        axis=(2, 3, 4, 5)
                    )

        output += self.biases.data.reshape(1, -1, 1, 1, 1)

        self.output = Tensor(output)
        return self.output

    def backward(self, dL_dout: Tensor, lr: float):
        batch_size, _, out_depth, out_height, out_width = dL_dout.data.shape

        dL_dfilters = np.zeros_like(self.filters.data)
        dL_dbiases = np.sum(dL_dout.data, axis=(0, 2, 3, 4), keepdims=True)
        dL_dinputs = np.zeros_like(self.inputs.data)

        if any(p > 0 for p in self.padding):
            padded_inputs = np.pad(self.inputs.data, ((0, 0), (0, 0),
                                                      (self.padding[0], self.padding[0]),
                                                      (self.padding[1], self.padding[1]),
                                                      (self.padding[2], self.padding[2])), mode='constant')
        else:
            padded_inputs = self.inputs.data

        for i in range(out_depth):
            for j in range(out_height):
                for k in range(out_width):
                    d_start = i * self.stride[0]
                    d_end = d_start + self.kernel_size[0]
                    h_start = j * self.stride[1]
                    h_end = h_start + self.kernel_size[1]
                    w_start = k * self.stride[2]
                    w_end = w_start + self.kernel_size[2]

                    dL_dfilters += np.sum(
                        padded_inputs[:, np.newaxis, :, d_start:d_end, h_start:h_end, w_start:w_end] *
                        dL_dout.data[:, :, i:i + 1, j:j + 1, k:k + 1],
                        axis=0
                    )
                    dL_dinputs[:, :, d_start:d_end, h_start:h_end, w_start:w_end] += np.sum(
                        self.filters.data[np.newaxis, :, :, :, :, :] *
                        dL_dout.data[:, :, i:i + 1, j:j + 1, k:k + 1],
                        axis=1
                    )

        if any(p > 0 for p in self.padding):
            dL_dinputs = dL_dinputs[:, :,
                         self.padding[0]:-self.padding[0],
                         self.padding[1]:-self.padding[1],
                         self.padding[2]:-self.padding[2]]

        self.filters.grad = dL_dfilters if self.filters.grad is None else self.filters.grad + dL_dfilters
        self.biases.grad = dL_dbiases if self.biases.grad is None else self.biases.grad + dL_dbiases

        self.filters.data -= lr * self.filters.grad
        self.biases.data -= lr * self.biases.grad

        self.inputs.grad = dL_dinputs if self.inputs.grad is None else self.inputs.grad + dL_dinputs
        self.inputs.backward_fn = lambda grad: grad + dL_dinputs if self.inputs.backward_fn is None else lambda \
                x: self.inputs.backward_fn(x) + dL_dinputs

        return Tensor(dL_dinputs)

    def get_params(self):
        return {
            'filters': self.filters.data,
            'biases': self.biases.data
        }

    def set_params(self, params):
        self.filters.data = params['filters']
        self.biases.data = params['biases']


class L1RegularizationLayer:
    """
    Parameters:
    -----------
    self layer: Layer
    self lambda_: lambda_
    """

    def __init__(self, layer, lambda_):
        self.layer = layer
        self.lambda_ = lambda_

    def forward(self, inputs: Tensor):
        return self.layer.forward(inputs)

    def backward(self, dL_dout: Tensor, lr: float):
        dL_dx = self.layer.backward(dL_dout, lr)

        l1_grad_gamma = self.lambda_ * np.sign(self.layer.gamma.data)
        l1_grad_beta = self.lambda_ * np.sign(self.layer.beta.data)

        self.layer.gamma.grad += l1_grad_gamma.reshape(
            self.layer.gamma.grad.shape)  # add gradient to +=  reshaped l1_grad
        self.layer.beta.grad += l1_grad_beta.reshape(self.layer.beta.grad.shape)

        self.layer.gamma.data -= lr * l1_grad_gamma.reshape(self.layer.gamma.data.shape)
        self.layer.beta.data -= lr * l1_grad_beta.reshape(self.layer.beta.data.shape)

        return dL_dx

    def get_params(self):
        return self.layer.get_params()

    def set_params(self, params):
        self.layer.set_params(params)


class L2RegularizationLayer:
    # The same as in L1 but with 2 dimensions
    def __init__(self, layer, lambda_):
        self.layer = layer
        self.lambda_ = lambda_

    def forward(self, inputs):
        return self.layer.forward(inputs)

    def backward(self, dL_dout, lr):
        dL_dx = self.layer.backward(dL_dout, lr)

        l2_grad_gamma = 2 * self.lambda_ * self.layer.gamma.data
        l2_grad_beta = 2 * self.lambda_ * self.layer.beta.data

        self.layer.gamma.grad += l2_grad_gamma.reshape(self.layer.gamma.grad.shape)
        self.layer.beta.grad += l2_grad_beta.reshape(self.layer.beta.grad.shape)

        self.layer.gamma.data -= lr * l2_grad_gamma.reshape(self.layer.gamma.data.shape)
        self.layer.beta.data -= lr * l2_grad_beta.reshape(self.layer.beta.data.shape)

        return dL_dx

    def get_params(self):
        return self.layer.get_params()

    def set_params(self, params):
        self.layer.set_params(params)


class L3RegularizationLayer:
    # The same as in L1 but with 3 dimensions
    def __init__(self, layer, lambda_):
        self.layer = layer
        self.lambda_ = lambda_

    def forward(self, inputs):
        return self.layer.forward(inputs)

    def backward(self, dL_dout, lr):
        dL_dx = self.layer.backward(dL_dout, lr)

        l3_grad_gamma = 3 * self.lambda_ * self.layer.gamma.data ** 2
        l3_grad_beta = 3 * self.lambda_ * self.layer.beta.data ** 2

        self.layer.gamma.grad += l3_grad_gamma.reshape(self.layer.gamma.grad.shape)
        self.layer.beta.grad += l3_grad_beta.reshape(self.layer.beta.grad.shape)

        self.layer.gamma.data -= lr * l3_grad_gamma.reshape(self.layer.gamma.data.shape)
        self.layer.beta.data -= lr * l3_grad_beta.reshape(self.layer.beta.data.shape)

        return dL_dx

    def get_params(self):
        return self.layer.get_params()

    def set_params(self, params):
        self.layer.set_params(params)


class BatchNormLayer:
    """
    Batch Normalization Layer for neural networks.

    Parameters:
    -----------
    num_features: int - The number of features in the input.
    eps: float - A small value to avoid division by zero.
    momentum: float - A momentum term for running mean/var updates.
    """

    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        self.gamma = Tensor(np.ones((1, num_features)))
        self.beta = Tensor(np.zeros((1, num_features)))

        self.running_mean = np.zeros((1, num_features))
        self.running_var = np.ones((1, num_features))

        self.training = True

    def forward(self, inputs: Tensor):
        self.inputs = inputs
        self.batch_size, _ = inputs.data.shape

        if self.training:
            mean = np.mean(inputs.data, axis=0)
            var = np.var(inputs.data, axis=0)

            # Update running statistics
            self.running_mean = self.momentum * mean + (1 - self.momentum) * self.running_mean
            self.running_var = self.momentum * var + (1 - self.momentum) * self.running_var
        else:
            mean = self.running_mean
            var = self.running_var

        # Ensure variance is not too small or zero for numerical stability
        var = np.maximum(var, self.eps)

        self.x_centered = inputs.data - mean
        self.x_norm = self.x_centered / np.sqrt(var + self.eps)

        outputs = self.gamma.data * self.x_norm + self.beta.data

        # Ensure no NaNs or Infs in outputs
        outputs = np.nan_to_num(outputs, nan=0.0, posinf=1.0, neginf=0.0)

        self.outputs = Tensor(outputs)
        return self.outputs

    def backward(self, dL_dout: Tensor, lr: float):
        dL_dgamma = np.sum(dL_dout.data * self.x_norm, axis=0)
        dL_dbeta = np.sum(dL_dout.data, axis=0)

        dL_dx_norm = dL_dout.data * self.gamma.data
        dL_dvar = np.sum(dL_dx_norm * self.x_centered * -0.5 * np.power(self.running_var + self.eps, -1.5), axis=0)

        dL_dmean = np.sum(dL_dx_norm * -1 / np.sqrt(self.running_var + self.eps), axis=0)
        dL_dmean += dL_dvar * np.mean(-2 * self.x_centered, axis=0)

        dL_dx = dL_dx_norm / np.sqrt(self.running_var + self.eps)
        dL_dx += dL_dvar * 2 * self.x_centered / self.batch_size
        dL_dx += dL_dmean / self.batch_size

        # Accumulate gradients for gamma and beta
        if self.gamma.grad is None:
            self.gamma.grad = dL_dgamma
        else:
            self.gamma.grad += dL_dgamma

        if self.beta.grad is None:
            self.beta.grad = dL_dbeta
        else:
            self.beta.grad += dL_dbeta

        # Define the backward function for inputs (if needed for further backpropagation)
        if self.inputs.backward_fn is None:
            self.inputs.backward_fn = lambda grad: grad + dL_dx
        else:
            prev_backward_fn = self.inputs.backward_fn
            self.inputs.backward_fn = lambda grad: prev_backward_fn(grad) + dL_dx

        # Update parameters
        self.gamma.data -= lr * self.gamma.grad
        self.beta.data -= lr * self.beta.grad

        return Tensor(dL_dx)

    def get_params(self):
        return {
            'gamma': self.gamma.data,
            'beta': self.beta.data,
            'running_mean': self.running_mean,
            'running_var': self.running_var
        }

    def set_params(self, params):
        self.gamma.data = params['gamma']
        self.beta.data = params['beta']
        self.running_mean = params['running_mean']
        self.running_var = params['running_var']


class InstanceNormLayer:
    """
    Parameters:
    -----------
    self num_features: num_features
    self eps: eps
    self gamma: gamma
    self.beta: beta
    self.running_mean: running_mean
    self.running_var: running_var
    self.training: training

    Explanations:
    ------------
    self.training: If True, the layer is in training mode, else in evaluation mode
    self.gamma and self.beta are learnable parameters
    self.momentum is a hyperparameter
    self.eps is a small value to avoid division by zero
    self.running_mean and self.running_var are running estimates of the mean and variance
    self.x_centered is the centered input
    self.x_norm is the normalized input
    self.outputs is the output of the layer
    self.num_features is the number of features in the input
    """

    def __init__(self, num_features, eps=1e-5):
        self.num_features = num_features
        self.eps = eps
        self.gamma = Tensor(np.ones((1, num_features)))
        self.beta = Tensor(np.zeros((1, num_features)))

    def forward(self, inputs: Tensor):
        self.inputs = inputs
        self.batch_size, _ = inputs.data.shape

        mean = np.mean(inputs.data, axis=1, keepdims=True)
        var = np.var(inputs.data, axis=1, keepdims=True)

        self.x_centered = inputs.data - mean
        self.x_norm = self.x_centered / np.sqrt(var + self.eps)
        outputs = self.gamma.data * self.x_norm + self.beta.data
        self.outputs = Tensor(outputs)
        return self.outputs

    def backward(self, dL_dout: Tensor, lr: float):
        dL_dgamma = np.sum(dL_dout.data * self.x_norm, axis=(0, 1), keepdims=True)
        dL_dbeta = np.sum(dL_dout.data, axis=(0, 1), keepdims=True)

        dL_dx_norm = dL_dout.data * self.gamma.data
        dL_dvar = np.sum(
            dL_dx_norm * self.x_centered * -0.5 * (self.inputs.data.var(axis=1, keepdims=True) + self.eps) ** (-3 / 2),
            axis=1, keepdims=True)
        dL_dmean = np.sum(dL_dx_norm * -1 / np.sqrt(self.inputs.data.var(axis=1, keepdims=True) + self.eps), axis=1,
                          keepdims=True)
        dL_dmean += dL_dvar * np.mean(-2 * self.x_centered, axis=1, keepdims=True)

        dL_dx = dL_dx_norm / np.sqrt(self.inputs.data.var(axis=1, keepdims=True) + self.eps)
        dL_dx += dL_dvar * 2 * self.x_centered / self.inputs.data.shape[1]
        dL_dx += dL_dmean / self.inputs.data.shape[1]

        self.gamma.grad = dL_dgamma if self.gamma.grad is None else self.gamma.grad + dL_dgamma
        self.beta.grad = dL_dbeta if self.beta.grad is None else self.beta.grad + dL_dbeta
        self.inputs.backward_fn = lambda grad: grad + dL_dx if self.inputs.backward_fn is None else lambda \
                x: self.inputs.backward_fn(x) + dL_dx

        self.gamma.data -= lr * self.gamma.grad
        self.beta.data -= lr * self.beta.grad

        return Tensor(dL_dx)

    def get_params(self):
        return {
            'gamma': self.gamma.data,
            'beta': self.beta.data
        }

    def set_params(self, params):
        self.gamma.data = params['gamma']
        self.beta.data = params['beta']


class DropoutLayer:
    """
    Parameters:
    -----------
    self rate: rate
    self.mask: mask
    self.inputs: inputs
    self.outputs: outputs
    self.training: training

    Explanations:
    ------------
    self.training: If True, the layer is in training mode, else in evaluation mode
    self.mask is the dropout mask
    self.outputs is the output of the layer
    """

    def __init__(self, rate=0.5):
        self.rate = rate

    def forward(self, inputs: Tensor, training=True):
        self.inputs = inputs
        if not training:
            return inputs
        self.mask = np.random.binomial(1, 1 - self.rate, size=inputs.data.shape) / (1 - self.rate)
        self.outputs = Tensor(inputs.data * self.mask)
        return self.outputs

    def backward(self, dL_dout: Tensor, lr: float):
        dL_dinputs = dL_dout.data * self.mask

        self.inputs.grad = dL_dinputs if self.inputs.grad is None else self.inputs.grad + dL_dinputs
        self.inputs.backward_fn = lambda grad: grad + dL_dinputs if self.inputs.backward_fn is None else lambda \
                x: self.inputs.backward_fn(x) + dL_dinputs

        return Tensor(dL_dinputs)

    def get_params(self):
        return None

    def set_params(self, params):
        pass


class CrossEntropyLoss:
    """
    Parameters:
    -----------
    self outputs: outputs
    self targets: targets

    """

    def forward(self, outputs: Tensor, targets: Tensor):
        samples = len(outputs.data)
        clipped_outputs = np.clip(outputs.data, 1e-12, 1 - 1e-12)
        correct_confidences = clipped_outputs[range(samples), targets.data.astype(int)]
        negative_log_likelihoods = -np.log(correct_confidences)
        loss = np.mean(negative_log_likelihoods)
        self.outputs = outputs
        self.targets = targets
        return loss

    def backward(self, dL_dloss: Tensor, lr: float):
        samples = len(self.outputs.data)
        clipped_outputs = np.clip(self.outputs.data, 1e-12, 1 - 1e-12)
        clipped_outputs[range(samples), self.targets.data.astype(int)] -= 1
        dL_doutputs = Tensor(clipped_outputs / samples)

        self.outputs.grad = dL_doutputs.data if self.outputs.grad is None else self.outputs.grad + dL_doutputs.data
        self.outputs.backward_fn = lambda grad: grad + dL_doutputs.data if self.outputs.backward_fn is None else lambda \
                x: self.outputs.backward_fn(x) + dL_doutputs.data

        return dL_doutputs


class LSTMLayer:
    """
    Parameters:
    -----------
    self input_size: input size
    self hidden_size: hidden size
    self Wf: weight for forget gate
    self Wi: weight for input gate
    self Wc: weight for cell gate
    self Wo: weight for output gate
    self bf: bias for forget gate
    self bi: bias for input gate
    self bc: bias for cell gate
    self bo: bias for output gate
    self hidden: hidden
    self cell: cell
    self outputs: output
    """

    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.Wf = Tensor(np.random.randn(input_size + hidden_size, hidden_size) * 0.01)
        self.Wi = Tensor(np.random.randn(input_size + hidden_size, hidden_size) * 0.01)
        self.Wc = Tensor(np.random.randn(input_size + hidden_size, hidden_size) * 0.01)
        self.Wo = Tensor(np.random.randn(input_size + hidden_size, hidden_size) * 0.01)

        self.bf = Tensor(np.zeros((1, hidden_size)))
        self.bi = Tensor(np.zeros((1, hidden_size)))
        self.bc = Tensor(np.zeros((1, hidden_size)))
        self.bo = Tensor(np.zeros((1, hidden_size)))

    def forward(self, inputs: Tensor, prev_hidden=None, prev_cell=None):
        if prev_hidden is None:
            prev_hidden = Tensor(np.zeros((inputs.data.shape[0], self.hidden_size)))
        if prev_cell is None:
            prev_cell = Tensor(np.zeros((inputs.data.shape[0], self.hidden_size)))

        self.inputs = inputs
        self.prev_hidden = prev_hidden
        self.prev_cell = prev_cell

        combined = np.concatenate((inputs.data, prev_hidden.data), axis=1)

        f = self.sigmoid(Tensor(np.dot(combined, self.Wf.data) + self.bf.data))

        i = self.sigmoid(Tensor(np.dot(combined, self.Wi.data) + self.bi.data))

        c_candidate = np.tanh(np.dot(combined, self.Wc.data) + self.bc.data)

        cell = f.data * prev_cell.data + i.data * c_candidate

        o = self.sigmoid(Tensor(np.dot(combined, self.Wo.data) + self.bo.data))

        hidden = o.data * np.tanh(cell)

        self.f, self.i, self.c_candidate, self.o = f, i, Tensor(c_candidate), o
        self.cell = Tensor(cell)
        self.hidden = Tensor(hidden)

        return self.hidden, self.cell

    def backward(self, dL_dh: Tensor, dL_dc: Tensor, lr: float):
        dL_do = dL_dh.data * np.tanh(self.cell.data)
        dL_dcell = dL_dc.data + dL_dh.data * self.o.data * (1 - np.tanh(self.cell.data) ** 2)

        dL_df = dL_dcell * self.prev_cell.data
        dL_di = dL_dcell * self.c_candidate.data
        dL_dc_candidate = dL_dcell * self.i.data

        dL_dWf = np.dot(np.concatenate((self.inputs.data, self.prev_hidden.data), axis=1).T,
                        dL_df * self.f.data * (1 - self.f.data))
        dL_dWi = np.dot(np.concatenate((self.inputs.data, self.prev_hidden.data), axis=1).T,
                        dL_di * self.i.data * (1 - self.i.data))
        dL_dWc = np.dot(np.concatenate((self.inputs.data, self.prev_hidden.data), axis=1).T,
                        dL_dc_candidate * (1 - self.c_candidate.data ** 2))
        dL_dWo = np.dot(np.concatenate((self.inputs.data, self.prev_hidden.data), axis=1).T,
                        dL_do * self.o.data * (1 - self.o.data))

        dL_dbf = np.sum(dL_df * self.f.data * (1 - self.f.data), axis=0, keepdims=True)
        dL_dbi = np.sum(dL_di * self.i.data * (1 - self.i.data), axis=0, keepdims=True)
        dL_dbc = np.sum(dL_dc_candidate * (1 - self.c_candidate.data ** 2), axis=0, keepdims=True)
        dL_dbo = np.sum(dL_do * self.o.data * (1 - self.o.data), axis=0, keepdims=True)

        self.Wf.grad = dL_dWf if self.Wf.grad is None else self.Wf.grad + dL_dWf
        self.Wi.grad = dL_dWi if self.Wi.grad is None else self.Wi.grad + dL_dWi
        self.Wc.grad = dL_dWc if self.Wc.grad is None else self.Wc.grad + dL_dWc
        self.Wo.grad = dL_dWo if self.Wo.grad is None else self.Wo.grad + dL_dWo
        self.bf.grad = dL_dbf if self.bf.grad is None else self.bf.grad + dL_dbf
        self.bi.grad = dL_dbi if self.bi.grad is None else self.bi.grad + dL_dbi
        self.bc.grad = dL_dbc if self.bc.grad is None else self.bc.grad + dL_dbc
        self.bo.grad = dL_dbo if self.bo.grad is None else self.bo.grad + dL_dbo

        dL_dprev_h = np.dot(dL_df * self.f.data * (1 - self.f.data), self.Wf.data[self.input_size:].T) + \
                     np.dot(dL_di * self.i.data * (1 - self.i.data), self.Wi.data[self.input_size:].T) + \
                     np.dot(dL_dc_candidate * (1 - self.c_candidate.data ** 2), self.Wc.data[self.input_size:].T) + \
                     np.dot(dL_do * self.o.data * (1 - self.o.data), self.Wo.data[self.input_size:].T)

        dL_dprev_c = dL_dcell * self.f.data

        self.prev_hidden.grad = dL_dprev_h if self.prev_hidden.grad is None else self.prev_hidden.grad + dL_dprev_h
        self.prev_hidden.backward_fn = lambda \
                grad: grad + dL_dprev_h if self.prev_hidden.backward_fn is None else lambda \
            x: self.prev_hidden.backward_fn(
            x) + dL_dprev_h

        self.prev_cell.grad = dL_dprev_c if self.prev_cell.grad is None else self.prev_cell.grad + dL_dprev_c
        self.prev_cell.backward_fn = lambda grad: grad + dL_dprev_c if self.prev_cell.backward_fn is None else lambda \
                x: self.prev_cell.backward_fn(x) + dL_dprev_c

        return Tensor(dL_dprev_h), Tensor(dL_dprev_c)

    def sigmoid(self, x: Tensor):
        return x.apply(lambda z: 1 / (1 + np.exp(-z)))

    def get_params(self):
        return {
            'Wf': self.Wf.data, 'Wi': self.Wi.data, 'Wc': self.Wc.data, 'Wo': self.Wo.data,
            'bf': self.bf.data, 'bi': self.bi.data, 'bc': self.bc.data, 'bo': self.bo.data
        }

    def set_params(self, params):
        self.Wf.data = params['Wf']
        self.Wi.data = params['Wi']
        self.Wc.data = params['Wc']
        self.Wo.data = params['Wo']
        self.bf.data = params['bf']
        self.bi.data = params['bi']
        self.bc.data = params['bc']
        self.bo.data = params['bo']


class GRULayer:
    # Simillar to LTSM layer
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.Wz = Tensor(np.random.randn(input_size + hidden_size, hidden_size) * 0.01)
        self.Wr = Tensor(np.random.randn(input_size + hidden_size, hidden_size) * 0.01)
        self.Wh = Tensor(np.random.randn(input_size + hidden_size, hidden_size) * 0.01)

        self.bz = Tensor(np.zeros((1, hidden_size)))
        self.br = Tensor(np.zeros((1, hidden_size)))
        self.bh = Tensor(np.zeros((1, hidden_size)))

    def forward(self, inputs: Tensor, prev_hidden=None):
        if prev_hidden is None:
            prev_hidden = Tensor(np.zeros((inputs.data.shape[0], self.hidden_size)))

        self.inputs = inputs
        self.prev_hidden = prev_hidden

        combined = np.concatenate((inputs.data, prev_hidden.data), axis=1)

        z = self.sigmoid(Tensor(np.dot(combined, self.Wz.data) + self.bz.data))

        r = self.sigmoid(Tensor(np.dot(combined, self.Wr.data) + self.br.data))

        h_candidate = np.tanh(
            np.dot(np.concatenate((inputs.data, r.data * prev_hidden.data), axis=1), self.Wh.data) + self.bh.data)

        hidden = (1 - z.data) * prev_hidden.data + z.data * h_candidate

        self.z, self.r, self.h_candidate = z, r, Tensor(h_candidate)
        self.hidden = Tensor(hidden)

        return self.hidden

    def backward(self, dL_dh: Tensor, lr: float):
        dL_dz = dL_dh.data * (self.h_candidate.data - self.prev_hidden.data)
        dL_dh_candidate = dL_dh.data * self.z.data
        dL_dr = np.dot(dL_dh_candidate * (1 - self.h_candidate.data ** 2),
                       self.Wh.data[self.input_size:].T) * self.prev_hidden.data

        dL_dWz = np.dot(np.concatenate((self.inputs.data, self.prev_hidden.data), axis=1).T,
                        dL_dz * self.z.data * (1 - self.z.data))
        dL_dWr = np.dot(np.concatenate((self.inputs.data, self.prev_hidden.data), axis=1).T,
                        dL_dr * self.r.data * (1 - self.r.data))
        dL_dWh = np.dot(np.concatenate((self.inputs.data, self.r.data * self.prev_hidden.data), axis=1).T,
                        dL_dh_candidate * (1 - self.h_candidate.data ** 2))

        dL_dbz = np.sum(dL_dz * self.z.data * (1 - self.z.data), axis=0, keepdims=True)
        dL_dbr = np.sum(dL_dr * self.r.data * (1 - self.r.data), axis=0, keepdims=True)
        dL_dbh = np.sum(dL_dh_candidate * (1 - self.h_candidate.data ** 2), axis=0, keepdims=True)

        self.Wz.grad = dL_dWz if self.Wz.grad is None else self.Wz.grad + dL_dWz
        self.Wr.grad = dL_dWr if self.Wr.grad is None else self.Wr.grad + dL_dWr
        self.Wh.grad = dL_dWh if self.Wh.grad is None else self.Wh.grad + dL_dWh
        self.bz.grad = dL_dbz if self.bz.grad is None else self.bz.grad + dL_dbz
        self.br.grad = dL_dbr if self.br.grad is None else self.br.grad + dL_dbr
        self.bh.grad = dL_dbh if self.bh.grad is None else self.bh.grad + dL_dbh

        dL_dprev_h = np.dot(dL_dz * self.z.data * (1 - self.z.data), self.Wz.data[self.input_size:].T) + \
                     np.dot(dL_dr * self.r.data * (1 - self.r.data), self.Wr.data[self.input_size:].T) + \
                     np.dot(dL_dh_candidate * (1 - self.h_candidate.data ** 2),
                            self.Wh.data[self.input_size:].T) * self.r.data + \
                     dL_dh.data * (1 - self.z.data)

        dL_dx = np.dot(dL_dz * self.z.data * (1 - self.z.data), self.Wz.data[:self.input_size].T) + \
                np.dot(dL_dr * self.r.data * (1 - self.r.data), self.Wr.data[:self.input_size].T) + \
                np.dot(dL_dh_candidate * (1 - self.h_candidate.data ** 2), self.Wh.data[:self.input_size].T)

        self.prev_hidden.grad = dL_dprev_h if self.prev_hidden.grad is None else self.prev_hidden.grad + dL_dprev_h
        self.prev_hidden.backward_fn = lambda \
                grad: grad + dL_dprev_h if self.prev_hidden.backward_fn is None else lambda \
            x: self.prev_hidden.backward_fn(
            x) + dL_dprev_h

        self.inputs.grad = dL_dx if self.inputs.grad is None else self.inputs.grad + dL_dx
        self.inputs.backward_fn = lambda grad: grad + dL_dx if self.inputs.backward_fn is None else lambda \
                x: self.inputs.backward_fn(x) + dL_dx

        return Tensor(dL_dx), Tensor(dL_dprev_h)

    def sigmoid(self, x: Tensor):
        return x.apply(lambda z: 1 / (1 + np.exp(-z)))

    def get_params(self):
        return {
            'Wz': self.Wz.data, 'Wr': self.Wr.data, 'Wh': self.Wh.data,
            'bz': self.bz.data, 'br': self.br.data, 'bh': self.bh.data
        }

    def set_params(self, params):
        self.Wz.data = params['Wz']
        self.Wr.data = params['Wr']
        self.Wh.data = params['Wh']
        self.bz.data = params['bz']
        self.br.data = params['br']
        self.bh.data = params['bh']


class RNNLayer:
    # Simmilar to GRULayer
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.Wh = Tensor(np.random.randn(hidden_size, hidden_size) * 0.01)
        self.Wx = Tensor(np.random.randn(input_size, hidden_size) * 0.01)
        self.b = Tensor(np.zeros((1, hidden_size)))

    def forward(self, inputs: Tensor, prev_hidden=None):
        if prev_hidden is None:
            prev_hidden = Tensor(np.zeros((inputs.data.shape[0], self.hidden_size)))

        self.inputs = inputs
        self.prev_hidden = prev_hidden

        hidden = np.tanh(np.dot(inputs.data, self.Wx.data) +
                         np.dot(prev_hidden.data, self.Wh.data) +
                         self.b.data)

        self.hidden = Tensor(hidden)
        return self.hidden

    def backward(self, dL_dh: Tensor, lr: float):
        dL_dWh = np.dot(self.prev_hidden.data.T, dL_dh.data * (1 - self.hidden.data ** 2))
        dL_dWx = np.dot(self.inputs.data.T, dL_dh.data * (1 - self.hidden.data ** 2))
        dL_db = np.sum(dL_dh.data * (1 - self.hidden.data ** 2), axis=0, keepdims=True)

        self.Wh.grad = dL_dWh if self.Wh.grad is None else self.Wh.grad + dL_dWh
        self.Wx.grad = dL_dWx if self.Wx.grad is None else self.Wx.grad + dL_dWx
        self.b.grad = dL_db if self.b.grad is None else self.b.grad + dL_db

        dL_dprev_h = np.dot(dL_dh.data * (1 - self.hidden.data ** 2), self.Wh.data.T)
        dL_dx = np.dot(dL_dh.data * (1 - self.hidden.data ** 2), self.Wx.data.T)

        self.prev_hidden.grad = dL_dprev_h if self.prev_hidden.grad is None else self.prev_hidden.grad + dL_dprev_h
        self.inputs.grad = dL_dx if self.inputs.grad is None else self.inputs.grad + dL_dx

        return Tensor(dL_dx), Tensor(dL_dprev_h)

    def get_params(self):
        return {
            'Wh': self.Wh.data,
            'Wx': self.Wx.data,
            'b': self.b.data
        }

    def set_params(self, params):
        self.Wh.data = params['Wh']
        self.Wx.data = params['Wx']
        self.b.data = params['b']


class DummyLayer:
    # A dummy layer for testing
    def forward(self, inputs):
        return inputs

    def backward(self, grad, lr):
        return grad

    def get_params(self):
        return {}

    def set_params(self, params):
        pass


class MaxPoolingLayer:
    """
    Paramaters:
    self inputs = the inputs
    self output = the output
    self pool_size = the pool size
    self stride = the stride

    Explanation:
    Pooling layer with max pooling
    """

    def __init__(self, pool_size=2, stride=2):
        self.pool_size = pool_size
        self.stride = stride

    def forward(self, inputs):
        batch_size, channels, height, width = inputs.data.shape
        pool_height, pool_width = self.pool_size, self.pool_size
        stride = self.stride

        out_height = (height - pool_height) // stride + 1
        out_width = (width - pool_width) // stride + 1

        output = np.zeros((batch_size, channels, out_height, out_width))

        for i in range(out_height):
            for j in range(out_width):
                h_start = i * stride
                h_end = h_start + pool_height
                w_start = j * stride
                w_end = w_start + pool_width
                output[:, :, i, j] = np.max(inputs.data[:, :, h_start:h_end, w_start:w_end], axis=(2, 3))

        self.inputs = inputs
        self.output = Tensor(output)
        return self.output

    def backward(self, dL_dout: Tensor, lr: float = None):
        dL_dinputs = np.zeros_like(self.inputs.data)

        new_h, new_w, c = dL_dout.data.shape
        for i in range(new_h):
            for j in range(new_w):
                for k in range(c):
                    pool_region = self.inputs.data[i * self.stride:i * self.stride + self.pool_size,
                                  j * self.stride:j * self.stride + self.pool_size, k]
                    max_val = np.max(pool_region)
                    for m in range(self.pool_size):
                        for n in range(self.pool_size):
                            if pool_region[m, n] == max_val:
                                dL_dinputs[i * self.stride + m, j * self.stride + n, k] = dL_dout.data[i, j, k]

        self.inputs.grad = dL_dinputs if self.inputs.grad is None else self.inputs.grad + dL_dinputs
        self.inputs.backward_fn = lambda grad: grad + dL_dinputs if self.inputs.backward_fn is None else lambda \
                x: self.inputs.backward_fn(x) + dL_dinputs

        return Tensor(dL_dinputs)

    def get_params(self):
        return None

    def set_params(self, params):
        pass


class AveragePoolingLayer:
    # Simmilar to MaxPoolingLayer but with avaerage pooling
    def __init__(self, pool_size=2, stride=2):
        self.pool_size = pool_size
        self.stride = stride

    def forward(self, inputs):
        batch_size, channels, height, width = inputs.data.shape
        pool_height, pool_width = self.pool_size, self.pool_size
        stride = self.stride

        out_height = (height - pool_height) // stride + 1
        out_width = (width - pool_width) // stride + 1

        output = np.zeros((batch_size, channels, out_height, out_width))

        for i in range(out_height):
            for j in range(out_width):
                h_start = i * stride
                h_end = h_start + pool_height
                w_start = j * stride
                w_end = w_start + pool_width
                output[:, :, i, j] = np.mean(inputs.data[:, :, h_start:h_end, w_start:w_end], axis=(2, 3))

        self.inputs = inputs
        self.output = Tensor(output)
        return self.output

    def backward(self, dL_dout: Tensor, lr: float = None):
        dL_dinputs = np.zeros_like(self.inputs.data)

        new_h, new_w, c = dL_dout.data.shape
        for i in range(new_h):
            for j in range(new_w):
                for k in range(c):
                    avg_val = dL_dout.data[i, j, k] / (self.pool_size * self.pool_size)
                    for m in range(self.pool_size):
                        for n in range(self.pool_size):
                            dL_dinputs[i * self.stride + m, j * self.stride + n, k] += avg_val

        self.inputs.grad = dL_dinputs if self.inputs.grad is None else self.inputs.grad + dL_dinputs
        self.inputs.backward_fn = lambda grad: grad + dL_dinputs if self.inputs.backward_fn is None else lambda \
                x: self.inputs.backward_fn(x) + dL_dinputs

        return Tensor(dL_dinputs)

    def get_params(self):
        return None

    def set_params(self, params):
        pass

class MSELoss:
    # Basic loss function
    def forward(self, outputs, targets):
        return np.mean((outputs.data - targets.data) ** 2)

    def backward(self, outputs, targets):
        return Tensor(2 * (outputs.data - targets.data) / targets.data.size)


class MeanSquaredLogarithmicError:
    """
    Parameters
    ----------
    epsilon : float
        Constant for numerical stability
    """

    def forward(self, outputs: Tensor, targets: Tensor):
        epsilon = 1e-8
        return np.mean((np.log(outputs.data + 1 + epsilon) - np.log(targets.data + 1 + epsilon)) ** 2)

    def backward(self, outputs: Tensor, targets: Tensor):
        epsilon = 1e-8
        grad = 2 * (np.log(outputs.data + 1 + epsilon) - np.log(targets.data + 1 + epsilon)) / (
                outputs.data + 1 + epsilon)
        return Tensor(grad / targets.data.size)


class MeanAbsolutePercentageError:
    """
    Parameters
    ----------
    epsilon : float
        Constant for numerical stability
    """

    def forward(self, outputs: Tensor, targets: Tensor):
        epsilon = 1e-8
        return np.mean(np.abs((targets.data - outputs.data) / (targets.data + epsilon)) * 100)

    def backward(self, outputs: Tensor, targets: Tensor):
        epsilon = 1e-8
        grad = -100 * np.sign(targets.data - outputs.data) / (targets.data + epsilon)
        return Tensor(grad / targets.data.size)


class MeanAbsoluteError:
    def forward(self, outputs: Tensor, targets: Tensor):
        return np.mean(np.abs(targets.data - outputs.data))

    def backward(self, outputs: Tensor, targets: Tensor):
        grad = np.sign(outputs.data - targets.data)
        return Tensor(grad / outputs.data.size)


class CosineSimilarityLoss:
    """
    Parameters
    ----------
    dot_product : float
        Dot product of outputs and targets
    outputs_norm : float
        Norm of outputs
    targets_norm : float
        Norm of targets
    cosine_similarity : float
        Cosine similarity of outputs and targets
    """

    def forward(self, outputs: Tensor, targets: Tensor):
        dot_product = np.sum(outputs.data * targets.data, axis=1)
        outputs_norm = np.linalg.norm(outputs.data, axis=1)
        targets_norm = np.linalg.norm(targets.data, axis=1)
        cosine_similarity = dot_product / (outputs_norm * targets_norm + 1e-8)
        return np.mean(1 - cosine_similarity)

    def backward(self, outputs: Tensor, targets: Tensor):
        dot_product = np.sum(outputs.data * targets.data, axis=1, keepdims=True)
        outputs_norm = np.linalg.norm(outputs.data, axis=1, keepdims=True)
        targets_norm = np.linalg.norm(targets.data, axis=1, keepdims=True)

        grad = (targets.data / (outputs_norm * targets_norm + 1e-8)) - \
               (outputs.data * dot_product / (outputs_norm ** 3 * targets_norm + 1e-8))

        return Tensor(-grad / targets.data.shape[0])


class PaddingLayer:
    """
    Parameters:
    -----------
    self.padding : Union[int, Tuple[int, int], Tuple[int, int, int, int]] a padding attribute
    """

    def __init__(self, padding: Union[int, Tuple[int, int], Tuple[int, int, int, int]]):
        if isinstance(padding, int):
            self.padding = ((padding, padding), (padding, padding))
        elif len(padding) == 2:
            self.padding = ((padding[0], padding[0]), (padding[1], padding[1]))
        elif len(padding) == 4:
            self.padding = ((padding[0], padding[1]), (padding[2], padding[3]))
        else:
            raise ValueError(
                "Invalid padding format. Use int, (pad_h, pad_w) or (pad_top, pad_bottom, pad_left, pad_right)")

    def forward(self, inputs: Tensor) -> Tensor:
        raise NotImplementedError

    def backward(self, dL_dout: Tensor, lr: float = None) -> Tensor:
        raise NotImplementedError

    def get_params(self):
        return None

    def set_params(self, params):
        pass


class ZeroPaddingLayer(PaddingLayer):
    # The same like in PaddingLayer but with paddings set to 0
    def forward(self, inputs: Tensor) -> Tensor:
        self.inputs = inputs
        padded = np.pad(inputs.data, ((0, 0), (0, 0)) + self.padding, mode='constant', constant_values=0)
        return Tensor(padded)

    def backward(self, dL_dout: Tensor, lr: float = None) -> Tensor:
        pad_top, pad_bottom = self.padding[0]
        pad_left, pad_right = self.padding[1]
        dL_dinputs = dL_dout.data[:, :, pad_top:-pad_bottom, pad_left:-pad_right]

        self.inputs.grad = dL_dinputs if self.inputs.grad is None else self.inputs.grad + dL_dinputs
        self.inputs.backward_fn = lambda grad: grad + dL_dinputs if self.inputs.backward_fn is None else lambda \
                x: self.inputs.backward_fn(x) + dL_dinputs

        return Tensor(dL_dinputs)


class ReflectionPaddingLayer(PaddingLayer):
    # The same like in PaddingLayer but with paddings reflected
    def forward(self, inputs: Tensor) -> Tensor:
        self.inputs = inputs
        padded = np.pad(inputs.data, ((0, 0), (0, 0)) + self.padding, mode='reflect')
        return Tensor(padded)

    def backward(self, dL_dout: Tensor, lr: float = None) -> Tensor:
        pad_top, pad_bottom = self.padding[0]
        pad_left, pad_right = self.padding[1]
        dL_dinputs = dL_dout.data[:, :, pad_top:-pad_bottom, pad_left:-pad_right]

        self.inputs.grad = dL_dinputs if self.inputs.grad is None else self.inputs.grad + dL_dinputs
        self.inputs.backward_fn = lambda grad: grad + dL_dinputs if self.inputs.backward_fn is None else lambda \
                x: self.inputs.backward_fn(x) + dL_dinputs

        return Tensor(dL_dinputs)


class LinearActivationLayer:
    # Activation layer for linear activation, linear algebras
    def forward(self, inputs: Tensor):
        self.inputs = inputs
        return inputs

    def backward(self, dL_dout: Tensor, lr: float):
        dL_dinputs = dL_dout.data

        self.inputs.grad = dL_dinputs if self.inputs.grad is None else self.inputs.grad + dL_dinputs
        self.inputs.backward_fn = lambda grad: grad + dL_dinputs if self.inputs.backward_fn is None else lambda \
                x: self.inputs.backward_fn(x) + dL_dinputs

        return Tensor(dL_dinputs)

    def get_params(self):
        return None

    def set_params(self, params):
        pass


class BilinearLayer:
    """
    Parameters:
    -----------
    in1_features : int
    in2_features : int
    out_features : int
    self.weight : Tensor
    self.bias : Tensor
    Explanations :
    -------------
    in1_features: input size of the first input
    in2_features: input size of the second input
    out_features: output size
    self.weight: weight matrix
    self.bias: bias
    """

    def __init__(self, in1_features, in2_features, out_features):
        self.in1_features = in1_features
        self.in2_features = in2_features
        self.out_features = out_features

        self.weight = Tensor(
            np.random.randn(out_features, in1_features, in2_features) / np.sqrt(in1_features * in2_features))
        self.bias = Tensor(np.zeros(out_features))

    def forward(self, input1: Tensor, input2: Tensor):
        self.input1 = input1
        self.input2 = input2
        output = np.einsum('bi,bj,oij->bo', input1.data, input2.data, self.weight.data) + self.bias.data
        return Tensor(output)

    def backward(self, dL_dout: Tensor, lr: float):
        dL_dw = np.einsum('bo,bi,bj->oij', dL_dout.data, self.input1.data, self.input2.data)
        dL_db = np.sum(dL_dout.data, axis=0)
        dL_dinput1 = np.einsum('bo,oij,bj->bi', dL_dout.data, self.weight.data, self.input2.data)
        dL_dinput2 = np.einsum('bo,oij,bi->bj', dL_dout.data, self.weight.data, self.input1.data)

        self.weight.data -= lr * dL_dw
        self.bias.data -= lr * dL_db

        self.input1.grad = dL_dinput1 if self.input1.grad is None else self.input1.grad + dL_dinput1
        self.input2.grad = dL_dinput2 if self.input2.grad is None else self.input2.grad + dL_dinput2

        self.input1.backward_fn = lambda grad: grad + dL_dinput1 if self.input1.backward_fn is None else lambda \
                x: self.input1.backward_fn(x) + dL_dinput1
        self.input2.backward_fn = lambda grad: grad + dL_dinput2 if self.input2.backward_fn is None else lambda \
                x: self.input2.backward_fn(x) + dL_dinput2

        return Tensor(dL_dinput1), Tensor(dL_dinput2)

    def get_params(self):
        return [self.weight, self.bias]

    def set_params(self, params):
        self.weight, self.bias = params


class TanhActivationLayer:
    """
    Parameters:
    -----------
    inputs : Tensor
    output : Tensor
    dl_dinputs : Tensor
    dl_doutputs : Tensor
    Explanations :
    -------------
    inputs: input tensor
    output: output tensor
    dl_dinputs: gradient of the loss with respect to the inputs
    dl_doutputs: gradient of the loss with respect to the outputs
    """

    def forward(self, inputs: Tensor):
        self.inputs = inputs
        self.output = Tensor(np.tanh(inputs.data))
        return self.output

    def backward(self, dL_dout: Tensor, lr: float):
        dL_dinputs = dL_dout.data * (1 - np.tanh(self.inputs.data) ** 2)

        if self.inputs.requires_grad:
            self.inputs.grad = dL_dinputs if self.inputs.grad is None else self.inputs.grad + dL_dinputs
            if self.inputs.backward_fn is None:
                self.inputs.backward_fn = lambda grad: grad + dL_dinputs
            else:
                old_backward_fn = self.inputs.backward_fn
                self.inputs.backward_fn = lambda grad: old_backward_fn(grad) + dL_dinputs

        return Tensor(dL_dinputs)

    def get_params(self):
        return None

    def set_params(self, params):
        pass


class HardTanhActivationLayer:
    """
    Parameters:
    -----------
    inputs : Tensor
    output : Tensor
    dl_dinputs : Tensor
    dl_doutputs : Tensor
    Explanations :
    -------------
    inputs: input tensor
    output: output tensor
    dl_dinputs: gradient of the loss with respect to the inputs
    dl_doutputs: gradient of the loss with respect to the outputs
    """

    def __init__(self):
        self.inputs: Optional[Tensor] = None
        self.output: Optional[Tensor] = None
        self.dl_dinputs: Optional[Tensor] = None
        self.dl_doutputs: Optional[Tensor] = None

    def forward(self, inputs: Tensor) -> Tensor:
        self.inputs = inputs
        self.output = Tensor(np.clip(inputs.data, -1, 1))
        return self.output

    def backward(self, dL_dout: Tensor, lr: float) -> Tensor:
        dL_dinputs = dL_dout.data * np.logical_and(self.inputs.data >= -1, self.inputs.data <= 1).astype(float)
        if self.inputs.requires_grad:
            self.inputs.grad = dL_dinputs if self.inputs.grad is None else self.inputs.grad + dL_dinputs
            if self.inputs.backward_fn is None:
                self.inputs.backward_fn = lambda grad: grad + dL_dinputs
            else:
                old_backward_fn = self.inputs.backward_fn
                self.inputs.backward_fn = lambda grad: old_backward_fn(grad) + dL_dinputs
        return Tensor(dL_dinputs)

    def get_params(self) -> None:
        return None

    def set_params(self, params) -> None:
        pass


class ScaledDotProductAttention:
    """
    Parameters:
    -----------
    d_model : int
    attention_scores : Tensor
    attention_weights : Tensor
    Explanations :
    -------------
    d_model: dimension of the model
    attention_scores: attention scores
    attention_weights: attention weights
    """

    def __init__(self, d_model):
        self.d_model = d_model
        self.scale = np.sqrt(d_model)

    def forward(self, Q: Tensor, K: Tensor, V: Tensor, mask=None):
        self.Q, self.K, self.V = Q, K, V

        attention_scores = Q.dot(K.transpose())
        attention_scores.data /= self.scale

        if mask is not None:
            attention_scores.data += (mask.data * -1e9)

        attention_weights = self.softmax(attention_scores)
        self.attention_weights = attention_weights

        output = attention_weights.dot(V)
        return output

    def backward(self, dL_dout: Tensor, lr: float):
        dL_dV = self.attention_weights.transpose().dot(dL_dout)
        dL_dattention_weights = dL_dout.dot(self.V.transpose())
        dL_dattention_scores = dL_dattention_weights * (self.attention_weights * (1 - self.attention_weights.data))
        dL_dattention_scores = dL_dattention_scores * (1 / self.scale)
        dL_dQ = dL_dattention_scores.dot(self.K)
        dL_dK = dL_dattention_scores.transpose().dot(self.Q)

        self.Q.grad = dL_dQ.data if self.Q.grad is None else self.Q.grad + dL_dQ.data
        self.K.grad = dL_dK.data if self.K.grad is None else self.K.grad + dL_dK.data
        self.V.grad = dL_dV.data if self.V.grad is None else self.V.grad + dL_dV.data

        return dL_dQ, dL_dK, dL_dV

    def softmax(self, x: Tensor):
        exp_x = np.exp(x.data - np.max(x.data, axis=-1, keepdims=True))
        return Tensor(exp_x / np.sum(exp_x, axis=-1, keepdims=True))

    def get_params(self):
        return None

    def set_params(self, params):
        pass


class Embedding:
    """
    Parameters:
    -----------
    vocab_size : int
    embedding_dim : int
    embeddings : Tensor
    input_indices : Tensor
    Explanations :
    -------------
    vocab_size: size of the vocabulary
    embedding_dim: dimension of the embedding
    embeddings: embedding matrix
    input_indices: input indices
    """

    def __init__(self, vocab_size: int, embedding_dim: int):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.embeddings = Tensor(np.random.randn(vocab_size, embedding_dim) * 0.01)
        self.input_indices = None

    def forward(self, input_indices: Tensor) -> Tensor:
        self.input_indices = input_indices
        return Tensor(self.embeddings.data[input_indices.data])

    def backward(self, output_gradient: Tensor, lr: float) -> None:
        embedding_gradient = np.zeros_like(self.embeddings.data)
        np.add.at(embedding_gradient, self.input_indices.data, output_gradient.data)
        self.embeddings.data -= lr * embedding_gradient

    def get_params(self) -> Tuple[np.ndarray]:
        return (self.embeddings.data,)

    def set_params(self, params: Tuple[np.ndarray]) -> None:
        self.embeddings.data = params[0]


class PairwiseDistance:
    def __init__(self, p=2, eps=1e-6, keepdim=False):
        """
        Initialize the PairwiseDistance layer.

        Args:
        p (int): The norm degree for pairwise distance. Default is 2 (Euclidean distance).
        eps (float): Small value to avoid division by zero. Default is 1e-6.
        keepdim (bool): Whether to keep the same dimensions as input. Default is False.
        """
        self.p = p
        self.eps = eps
        self.keepdim = keepdim
        self.diff = None
        self.norm = None
        self.x1 = None
        self.x2 = None

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        """
        Compute the pairwise distances between x1 and x2.

        Args:
        x1 (Tensor): First input tensor of shape (N, D)
        x2 (Tensor): Second input tensor of shape (M, D)

        Returns:
        Tensor: Pairwise distances of shape (N, M) or (N, M, 1) if keepdim is True
        """
        self.x1 = x1
        self.x2 = x2
        self.diff = x1.data[:, None, :] - x2.data[None, :, :]
        self.norm = np.power(np.abs(self.diff) + self.eps, self.p)
        output = np.power(np.sum(self.norm, axis=-1), 1 / self.p)

        if self.keepdim:
            output = np.expand_dims(output, -1)

        return Tensor(output)

    def backward(self, grad_output: Tensor, lr: float) -> Tuple[Tensor, Tensor]:
        """
        Compute the gradient of the pairwise distance.

        Args:
        grad_output (Tensor): Gradient of the loss with respect to the output of this layer
        lr (float): Learning rate (not used in this layer, but kept for consistency)

        Returns:
        Tuple[Tensor, Tensor]: Gradients with respect to x1 and x2
        """
        if self.keepdim:
            grad_output_data = grad_output.data.squeeze(-1)
        else:
            grad_output_data = grad_output.data

        grad_output_expanded = grad_output_data[:, :, None]
        dist = np.power(np.sum(self.norm, axis=-1), 1 / self.p - 1)
        dist_expanded = dist[:, :, None]

        grad_dist = self.p * np.power(np.abs(self.diff) + self.eps, self.p - 1) * np.sign(self.diff)
        grad = grad_output_expanded * dist_expanded * grad_dist / self.norm.sum(axis=-1, keepdims=True)

        grad_x1 = grad.sum(axis=1)
        grad_x2 = -grad.sum(axis=0)

        return Tensor(grad_x1), Tensor(grad_x2)

    def get_params(self) -> Tuple:
        return tuple()

    def set_params(self, params: Tuple) -> None:
        pass


class Encoder:
    """
    Params:
    input_dim: int
    hidden_dims: List[int]
    output_dim: int
    """

    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int):
        self.layers = []
        dims = [input_dim] + hidden_dims + [output_dim]

        for i in range(len(dims) - 1):
            self.layers.append({
                'W': Tensor(np.random.randn(dims[i], dims[i + 1]) * 0.01),
                'b': Tensor(np.zeros((1, dims[i + 1])))
            })

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers[:-1]:
            x = x.dot(layer['W']).__add__(layer['b']).apply(lambda x: np.maximum(0, x))

        # Last layer without activation
        x = x.dot(self.layers[-1]['W']).__add__(self.layers[-1]['b'])
        return x

    def parameters(self) -> List[Tensor]:
        return [param for layer in self.layers for param in layer.values()]


class Decoder:
    """
    Params:
    input_dim: int
    hidden_dims: List[int]
    output_dim: int
    Explanations:
    input_dim: input dimension of the data
    hidden_dims: list of hidden dimensions
    output_dim: output dimension of the data
    """

    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int):
        self.layers = []
        dims = [input_dim] + hidden_dims + [output_dim]
        for i in range(len(dims) - 1):
            self.layers.append({
                'W': Tensor(np.random.randn(dims[i], dims[i + 1]) * 0.01),
                'b': Tensor(np.zeros((1, dims[i + 1])))
            })

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers[:-1]:
            x = x.dot(layer['W']).__add__(layer['b']).apply(lambda x: np.maximum(0, x))  # ReLU activation
        # Last layer with sigmoid activation for output between 0 and 1
        x = x.dot(self.layers[-1]['W']).__add__(self.layers[-1]['b']).apply(lambda x: 1 / (1 + np.exp(-x)))
        return x

    def parameters(self) -> List[Tensor]:
        return [param for layer in self.layers for param in layer.values()]


class TransformerEncoder:
    """
    Params:
    input_dim: int
    num_heads: int
    ff_dim: int
    num_layers: int
    """

    def __init__(self, input_dim: int, num_heads: int, ff_dim: int, num_layers: int):
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.num_layers = num_layers

        self.layers = []
        for _ in range(num_layers):
            self.layers.append({
                'attention': MultiHeadAttention(input_dim, num_heads),
                'norm1': LayerNorm(input_dim),
                'ff': FeedForward(input_dim, ff_dim),
                'norm2': LayerNorm(input_dim)
            })

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            # Self-attention
            attention_output = layer['attention'].forward(x, x, x)
            x = x + attention_output
            x = layer['norm1'].forward(x)

            # Feed-forward
            ff_output = layer['ff'].forward(x)
            x = x + ff_output
            x = layer['norm2'].forward(x)

        return x

    def parameters(self) -> List[Tensor]:
        return [param for layer in self.layers for module in layer.values() for param in module.parameters()]


class TransformerDecoder:
    """
    Params:
    input_dim: int
    num_heads: int
    ff_dim: int
    num_layers: int
    output_dim: int
    """

    def __init__(self, input_dim: int, num_heads: int, ff_dim: int, num_layers: int, output_dim: int):
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.num_layers = num_layers
        self.output_dim = output_dim

        self.layers = []
        for _ in range(num_layers):
            self.layers.append({
                'self_attention': MultiHeadAttention(input_dim, num_heads),
                'norm1': LayerNorm(input_dim),
                'cross_attention': MultiHeadAttention(input_dim, num_heads),
                'norm2': LayerNorm(input_dim),
                'ff': FeedForward(input_dim, ff_dim),
                'norm3': LayerNorm(input_dim)
            })

        self.output_layer = Linear(input_dim, output_dim)

    def forward(self, x: Tensor, encoder_output: Tensor) -> Tensor:
        for layer in self.layers:
            # Self-attention
            self_attention_output = layer['self_attention'].forward(x, x, x)
            x = x + self_attention_output
            x = layer['norm1'].forward(x)

            # Cross-attention
            cross_attention_output = layer['cross_attention'].forward(x, encoder_output, encoder_output)
            x = x + cross_attention_output
            x = layer['norm2'].forward(x)

            # Feed-forward
            ff_output = layer['ff'].forward(x)
            x = x + ff_output
            x = layer['norm3'].forward(x)

        return self.output_layer.forward(x)

    def parameters(self) -> List[Tensor]:
        params = [param for layer in self.layers for module in layer.values() for param in module.parameters()]
        params.extend(self.output_layer.parameters())
        return params


class MultiHeadAttention:
    """
    Params:
    input_dim: int
    num_heads: int
    head_dim = input_dim // num_heads
    Explanations:
    - input_dim: Dimensionality of the input tensor
    - num_heads: Number of attention heads
    - head_dim: Dimensionality of each attention head
    """

    def __init__(self, input_dim: int, num_heads: int):
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads

        # Initialize weights
        self.W_q = Tensor(np.random.randn(input_dim, input_dim) * np.sqrt(2.0 / (2 * input_dim)))
        self.W_k = Tensor(np.random.randn(input_dim, input_dim) * np.sqrt(2.0 / (2 * input_dim)))
        self.W_v = Tensor(np.random.randn(input_dim, input_dim) * np.sqrt(2.0 / (2 * input_dim)))
        self.W_o = Tensor(np.random.randn(input_dim, input_dim) * np.sqrt(2.0 / (2 * input_dim)))

    def split_heads(self, x: np.ndarray) -> np.ndarray:
        batch_size, seq_length, _ = x.shape
        return x.reshape(batch_size, seq_length, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

    def forward(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        batch_size, seq_length, _ = query.data.shape

        # Compute Q, K, V
        q = self.split_heads(np.dot(query.data, self.W_q.data))
        k = self.split_heads(np.dot(key.data, self.W_k.data))
        v = self.split_heads(np.dot(value.data, self.W_v.data))

        # Compute attention scores
        attention_scores = np.matmul(q, k.transpose(0, 1, 3, 2)) / np.sqrt(self.head_dim)

        # Compute attention probabilities
        attention_probs = np.exp(attention_scores - np.max(attention_scores, axis=-1, keepdims=True))
        attention_probs /= np.sum(attention_probs, axis=-1, keepdims=True)

        # Compute context
        context = np.matmul(attention_probs, v)

        # Reshape and apply final linear transformation
        context = context.transpose(0, 2, 1, 3).reshape(batch_size, seq_length, self.input_dim)
        output = np.dot(context, self.W_o.data)

        return Tensor(output)

    def parameters(self) -> List[Tensor]:
        return [self.W_q, self.W_k, self.W_v, self.W_o]


class FeedForward:
    """
    Params:
    W1 = np.random.randn(input_dim, hidden_dim) * 0.01
    b1 = np.zeros((1, hidden_dim))
    W2 = np.random.randn(hidden_dim, input_dim) * 0.01
    b2 = np.zeros((1, input_dim))
    Explanations:
    - W1: Weight matrix for the first linear transformation
    - b1: Bias vector for the first linear transformation
    - W2: Weight matrix for the second linear transformation
    - b2: Bias vector for the second linear transformation
    """

    def __init__(self, input_dim: int, hidden_dim: int):
        self.W1 = Tensor(np.random.randn(input_dim, hidden_dim) * 0.01)
        self.b1 = Tensor(np.zeros((1, hidden_dim)))
        self.W2 = Tensor(np.random.randn(hidden_dim, input_dim) * 0.01)
        self.b2 = Tensor(np.zeros((1, input_dim)))

    def forward(self, x: Tensor) -> Tensor:
        hidden = x.dot(self.W1).__add__(self.b1).apply(lambda x: np.maximum(0, x))
        return hidden.dot(self.W2).__add__(self.b2)

    def parameters(self) -> List[Tensor]:
        return [self.W1, self.b1, self.W2, self.b2]


class LayerNorm:
    """
    Params:
    dim: int
    gamma = np.ones((1, dim))
    beta = np.zeros((1, dim))
    eps = 1e-5
    Explanations:
    - dim: Dimensionality of the input tensor
    - gamma: Scale parameter
    - beta: Shift parameter
    - eps: Epsilon value
    """

    def __init__(self, dim: int):
        self.gamma = Tensor(np.ones((1, dim)))
        self.beta = Tensor(np.zeros((1, dim)))
        self.eps = 1e-5

    def forward(self, x: Tensor) -> Tensor:
        mean = np.mean(x.data, axis=-1, keepdims=True)
        var = np.var(x.data, axis=-1, keepdims=True)
        x_norm = (x.data - mean) / np.sqrt(var + self.eps)
        return Tensor(x_norm * self.gamma.data + self.beta.data)

    def parameters(self) -> List[Tensor]:
        return [self.gamma, self.beta]


class Linear:
    """
    Params:
    - input_dim: Dimensionality of the input tensor
    - output_dim: Dimensionality of the output tensor
    Explanations:
    - input_dim: Dimensionality of the input tensor
    - output_dim: Dimensionality of the output tensor
    """

    def __init__(self, input_dim: int, output_dim: int):
        self.W = Tensor(np.random.randn(input_dim, output_dim) * 0.01)
        self.b = Tensor(np.zeros((1, output_dim)))

    def forward(self, x: Tensor) -> Tensor:
        return x.dot(self.W).__add__(self.b)

    def parameters(self) -> List[Tensor]:
        return [self.W, self.b]


class MaxUnpoolingLayer:
    def __init__(self, pool_size=2, stride=2):
        self.pool_size = pool_size
        self.stride = stride

    def forward(self, inputs):
        batch_size, channels, height, width = inputs.data.shape
        unpooled_height = (height - 1) * self.stride + self.pool_size
        unpooled_width = (width - 1) * self.stride + self.pool_size

        output = np.zeros((batch_size, channels, unpooled_height, unpooled_width))

        for i in range(height):
            for j in range(width):
                h_start = i * self.stride
                w_start = j * self.stride
                output[:, :, h_start:h_start+self.pool_size, w_start:w_start+self.pool_size] = np.expand_dims(inputs.data[:, :, i, j], axis=(2, 3))

        self.inputs = inputs
        self.output = Tensor(output)
        return self.output


    def backward(self, dL_dout: Tensor, lr: float = None):
        dL_dinputs = np.zeros_like(self.inputs.data)

        batch_size, channels, height, width = self.inputs.data.shape
        for i in range(height):
            for j in range(width):
                h_start = i * self.stride
                w_start = j * self.stride
                dL_dinputs[:, :, i, j] = np.sum(dL_dout.data[:, :, h_start:h_start+self.pool_size, w_start:w_start+self.pool_size], axis=(2, 3))

        self.inputs.grad = dL_dinputs if self.inputs.grad is None else self.inputs.grad + dL_dinputs
        self.inputs.backward_fn = lambda grad: grad + dL_dinputs if self.inputs.backward_fn is None else lambda x: self.inputs.backward_fn(x) + dL_dinputs

        return Tensor(dL_dinputs)

    def get_params(self):
        return None

    def set_params(self, params):
        pass

class AverageUnpoolingLayer:
    def __init__(self, pool_size=2, stride=2):
        self.pool_size = pool_size
        self.stride = stride

    def forward(self, inputs):
        batch_size, channels, height, width = inputs.data.shape
        unpooled_height = (height - 1) * self.stride + self.pool_size
        unpooled_width = (width - 1) * self.stride + self.pool_size

        output = np.zeros((batch_size, channels, unpooled_height, unpooled_width))

        for i in range(height):
            for j in range(width):
                h_start = i * self.stride
                w_start = j * self.stride
                output[:, :, h_start:h_start+self.pool_size, w_start:w_start+self.pool_size] = np.expand_dims(inputs.data[:, :, i, j], axis=(2, 3)) / (self.pool_size * self.pool_size)

        self.inputs = inputs
        self.output = Tensor(output)
        return self.output

    def backward(self, dL_dout: Tensor, lr: float = None):
        dL_dinputs = np.zeros_like(self.inputs.data)

        batch_size, channels, height, width = self.inputs.data.shape
        for i in range(height):
            for j in range(width):
                h_start = i * self.stride
                w_start = j * self.stride
                dL_dinputs[:, :, i, j] = np.sum(dL_dout.data[:, :, h_start:h_start+self.pool_size, w_start:w_start+self.pool_size], axis=(2, 3)) / (self.pool_size * self.pool_size)

        self.inputs.grad = dL_dinputs if self.inputs.grad is None else self.inputs.grad + dL_dinputs
        self.inputs.backward_fn = lambda grad: grad + dL_dinputs if self.inputs.backward_fn is None else lambda x: self.inputs.backward_fn(x) + dL_dinputs

        return Tensor(dL_dinputs)

    def get_params(self):
        return None

    def set_params(self, params):
        pass


class NeuralNetwork:
    def __init__(self, temperature=1.0):
        self.layers = []
        self.hooks = {
            'pre_forward': [],
            'post_forward': [],
            'pre_backward': [],
            'post_backward': [],
            'pre_epoch': [],
            'post_epoch': []
        }
        self.temperature = temperature
        self.profiler = cProfile.Profile()
        self.is_profiling = False
        self.debug_mode = False
        self.layer_outputs = []
        self.gradients = []
        self.parameter_history = []
        self.logger = self._setup_logger()
        self.breakpoints = {}
        self.error_history = []


    def _setup_logger(self):
        logger = logging.getLogger('NeuralNetwork')
        logger.setLevel(logging.DEBUG)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def set_debug_mode(self, mode: bool):
        self.debug_mode = mode
        self.logger.info(f"Debug mode set to: {mode}")

    def set_breakpoint(self, method_name: str, condition: Callable = None):
        self.breakpoints[method_name] = condition
        self.logger.info(f"Breakpoint set for method: {method_name}")

    def remove_breakpoint(self, method_name: str):
        if method_name in self.breakpoints:
            del self.breakpoints[method_name]
            self.logger.info(f"Breakpoint removed for method: {method_name}")
        else:
            self.logger.warning(f"No breakpoint found for method: {method_name}")

    def _check_breakpoint(self, method_name: str, *args, **kwargs):
        if method_name in self.breakpoints:
            condition = self.breakpoints[method_name]
            if condition is None or condition(*args, **kwargs):
                self.logger.info(f"Breakpoint hit in method: {method_name}")
                pdb.set_trace()

    def add(self, layer):
        self._check_breakpoint('add', layer)
        self.layers.append(layer)

    def remove(self, index):
        self._check_breakpoint('remove', index)
        del self.layers[index]

    def add_hook(self, hook_type: str, hook_fn: Callable):
        if hook_type in self.hooks:
            self.hooks[hook_type].append(hook_fn)
        else:
            raise ValueError(f"Invalid hook type: {hook_type}")

    def remove_hook(self, hook_type: str, hook_fn: Callable):
        if hook_type in self.hooks and hook_fn in self.hooks[hook_type]:
            self.hooks[hook_type].remove(hook_fn)

    def _run_hooks(self, hook_type: str, *args, **kwargs):
        for hook in self.hooks[hook_type]:
            hook(*args, **kwargs)

    def start_profiling(self):
        self.profiler.enable()
        self.is_profiling = True

    def stop_profiling(self):
        self.profiler.disable()
        self.is_profiling = False

    def print_profile_stats(self, sort_by='cumulative', lines=20):
        if not self.is_profiling:
            print("Profiling was not started. Use start_profiling() first.")
            return

        s = io.StringIO()
        ps = pstats.Stats(self.profiler, stream=s).sort_stats(sort_by)
        ps.print_stats(lines)
        print(s.getvalue())

    def _log_layer_output(self, layer_index: int, output: 'Tensor'):
        if self.debug_mode:
            self.layer_outputs.append((layer_index, output.copy()))

    def _log_gradient(self, layer_index: int, gradient: 'Tensor'):
        if self.debug_mode:
            self.gradients.append((layer_index, gradient.copy()))

    def _log_parameters(self):
        if self.debug_mode:
            params = [layer.get_params() for layer in self.layers]
            self.parameter_history.append(params)

    def forward(self, inputs: 'Tensor'):
        self._check_breakpoint('forward', inputs)
        if self.is_profiling:
            return self._profiled_forward(inputs)
        else:
            return self._forward(inputs)

    def _profiled_forward(self, inputs: 'Tensor'):
        self.profiler.enable()
        result = self._forward(inputs)
        self.profiler.disable()
        return result

    def _forward(self, inputs: 'Tensor'):
        self._run_hooks('pre_forward', inputs)
        self.layer_outputs = []
        for i, layer in enumerate(self.layers):
            inputs = layer.forward(inputs)
            self._log_layer_output(i, inputs)
        inputs.data = inputs.data / self.temperature
        self._run_hooks('post_forward', inputs)
        return inputs

    def backward(self, loss_gradient: 'Tensor', lr: float):
        self._check_breakpoint('backward', loss_gradient, lr)
        if self.is_profiling:
            return self._profiled_backward(loss_gradient, lr)
        else:
            return self._backward(loss_gradient, lr)

    def _profiled_backward(self, loss_gradient: 'Tensor', lr: float):
        self.profiler.enable()
        result = self._backward(loss_gradient, lr)
        self.profiler.disable()
        return result

    def _backward(self, loss_gradient: 'Tensor', lr: float):
        self._run_hooks('pre_backward', loss_gradient, lr)
        self.gradients = []
        for i, layer in reversed(list(enumerate(self.layers))):
            loss_gradient = layer.backward(loss_gradient, lr)
            self._log_gradient(i, loss_gradient)
        self._run_hooks('post_backward', loss_gradient, lr)
        self._log_parameters()

    def train(self, inputs: 'Tensor', targets: 'Tensor', epochs: int, lr: float, optimizer, batch_size: int, loss_function):
        self._check_breakpoint('train', inputs, targets, epochs, lr, batch_size, loss_function)
        if self.is_profiling:
            return self._profiled_train(inputs, targets, epochs, lr, optimizer, batch_size, loss_function)
        else:
            return self._train(inputs, targets, epochs, lr, optimizer, batch_size, loss_function)

    def _profiled_train(self, inputs: 'Tensor', targets: 'Tensor', epochs: int, lr: float, optimizer, batch_size: int,
                        loss_function):
        self.profiler.enable()
        result = self._train(inputs, targets, epochs, lr, optimizer, batch_size, loss_function)
        self.profiler.disable()
        return result


    def _train(self, inputs: 'Tensor', targets: 'Tensor', epochs: int, lr, optimizer, batch_size: int,
               loss_function, clip_grad=None):
        num_samples = inputs.data.shape[0]
        num_batches = int(np.ceil(num_samples / batch_size))
        losses = []

        for epoch in range(epochs):
            self._run_hooks('pre_epoch', epoch, epochs)
            epoch_loss = 0.0
            epoch_start_time = time.time()

            # Shuffle the data at the beginning of each epoch
            indices = np.arange(num_samples)
            np.random.shuffle(indices)
            inputs.data = inputs.data[indices]
            targets.data = targets.data[indices]

            for batch in range(num_batches):
                batch_start = batch * batch_size
                batch_end = min(batch_start + batch_size, num_samples)
                batch_inputs = Tensor(inputs.data[batch_start:batch_end])
                batch_targets = Tensor(targets.data[batch_start:batch_end])

                # Forward pass
                outputs = self.forward(batch_inputs)

                # Compute loss
                loss = loss_function.forward(outputs, batch_targets)
                epoch_loss += float(np.sum(loss.data))

                # Backward pass
                loss_gradient = loss_function.backward(outputs, batch_targets)
                self.backward(loss_gradient, lr)

                # Gradient clipping (if specified)
                if clip_grad is not None:
                    self._clip_gradients(clip_grad)

                # Update parameters using the optimizer
                for i, layer in enumerate(self.layers):
                    if hasattr(layer, 'parameters'):
                        for key, param in layer.parameters.items():
                            grad = layer.gradients[key]
                            optimizer.update(param, grad, f"layer_{i}_{key}")


            avg_loss = epoch_loss / num_batches
            losses.append(avg_loss)

            self.parameter_history.append(self.get_parameter_history())

            epoch_end_time = time.time()

            self.logger.info(
                f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f} - Time: {epoch_end_time - epoch_start_time:.2f}s")
            self._run_hooks('post_epoch', epoch, epochs, avg_loss)

        return losses

    def _clip_gradients(self, max_norm):
        """Clip the gradients to a maximum norm."""
        total_norm = 0.0
        for param in self.parameters():
            if param.grad is not None:
                param_norm = np.linalg.norm(param.grad.data)
                total_norm += param_norm ** 2
        total_norm = np.sqrt(total_norm)

        if total_norm > max_norm:
            clip_coef = max_norm / (total_norm + 1e-6)
            for param in self.parameters():
                if param.grad is not None:
                    param.grad.data *= clip_coef

    def parameters(self):
        return []

    def save(self, file_path: str):
        params = [layer.get_params() for layer in self.layers]
        with open(file_path, 'wb') as f:
            pickle.dump((params, self.temperature), f)

    def load(self, file_path: str):
        with open(file_path, 'rb') as f:
            params, self.temperature = pickle.load(f)
        for layer, param in zip(self.layers, params):
            layer.set_params(param)

    def save_weights(self, file_path: str):
        params = [layer.get_params() for layer in self.layers]
        with open(file_path, 'wb') as f:
            pickle.dump(params, f)

    def load_weights(self, file_path: str):
        with open(file_path, 'rb') as f:
            params = pickle.load(f)
        for layer, param in zip(self.layers, params):
            layer.set_params(param)

    def set_temperature(self, temperature: float):
        self.temperature = temperature

    def plot_loss(self, losses, title="Training Loss", xlabel="Epoch", ylabel="Loss"):
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(losses) + 1), losses)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.show()

    def get_layer_outputs(self):
        return self.layer_outputs

    def get_gradients(self):
        return self.gradients

    def get_parameter_history(self):
        return [param.data.copy() for param in self.parameters()]

    def plot_parameter_changes(self):
        self._check_breakpoint('plot_parameter_changes')
        if not self.parameter_history:
            self.logger.warning("No parameter history available. Make sure debug mode is enabled.")
            return

        num_layers = len(self.parameter_history[0])
        if num_layers == 0:
            self.logger.warning("No layers to plot. Ensure your model has parameters.")
            return

        fig, axes = plt.subplots(num_layers, 1, figsize=(10, 5 * num_layers))

        # Ensure axes is always a list, even if there's only one subplot
        if num_layers == 1:
            axes = [axes]

        for i in range(num_layers):
            layer_params = [params[i] for params in self.parameter_history if i < len(params)]

            def extract_numerics(obj):
                if isinstance(obj, (int, float, np.number)):
                    return [obj]
                elif isinstance(obj, (list, tuple, np.ndarray)):
                    return [item for sublist in map(extract_numerics, obj) for item in sublist]
                elif isinstance(obj, dict):
                    return [item for sublist in map(extract_numerics, obj.values()) for item in sublist]
                elif hasattr(obj, 'data'):
                    return extract_numerics(obj.data)
                else:
                    return []

            # Extract numeric values and compute mean for each parameter set
            plottable_params = []
            for param in layer_params:
                numerics = extract_numerics(param)
                if numerics:
                    plottable_params.append(np.mean(numerics))
                else:
                    plottable_params.append(np.nan)  # Use NaN for empty sets

            # Remove NaN values before plotting
            valid_params = [p for p in plottable_params if not np.isnan(p)]

            if valid_params:
                axes[i].plot(valid_params)
                axes[i].set_title(f"Layer {i + 1} Parameters")
                axes[i].set_xlabel("Training Step")
                axes[i].set_ylabel("Average Parameter Value")
            else:
                axes[i].text(0.5, 0.5, "No valid data to plot",
                             horizontalalignment='center', verticalalignment='center')

        plt.tight_layout()
        plt.show()

    def analyze_gradients(self):
        self._check_breakpoint('analyze_gradients')
        if not self.gradients:
            self.logger.warning("No gradients available. Make sure debug mode is enabled.")
            return

        for layer_index, gradient in self.gradients:
            mean = np.mean(gradient.data)
            std = np.std(gradient.data)
            max_val = np.max(gradient.data)
            min_val = np.min(gradient.data)
            self.logger.info(
                f"Layer {layer_index + 1} Gradient - Mean: {mean:.4f}, Std: {std:.4f}, Max: {max_val:.4f}, Min: {min_val:.4f}")

    def set_loss_function(self, loss_function):
        self.loss_function = loss_function

    def deploy(self, host='0.0.0.0', port=5000):
        self._check_breakpoint('deploy', host, port)
        self.app = Flask(__name__)
        self._setup_routes()
        self.app.run(host=host, port=port)

    def _setup_routes(self):
        @self.app.route('/predict', methods=['POST'])
        def predict():
            data = request.json
            if 'input' not in data:
                return jsonify({'error': 'No input provided'}), 400

            input_data = Tensor(np.array(data['input']))
            output = self.forward(input_data)
            return jsonify({'prediction': output.data.tolist()})

        @self.app.route('/train', methods=['POST'])
        def train():
            data = request.json
            if not all(key in data for key in ['inputs', 'targets', 'epochs', 'lr', 'batch_size']):
                return jsonify({'error': 'Missing required parameters'}), 400

            inputs = Tensor(np.array(data['inputs']))
            targets = Tensor(np.array(data['targets']))
            epochs = int(data['epochs'])
            lr = float(data['lr'])
            batch_size = int(data['batch_size'])
            optimizer = data.get('optimizer', 'SGDOptimizer')

            losses = self.train(inputs, targets, epochs, lr, optimizer, batch_size, loss_function=self.loss_function)
            return jsonify({'losses': losses})

        @self.app.route('/save_model', methods=['POST'])
        def save_model():
            data = request.json
            if 'file_path' not in data:
                return jsonify({'error': 'No file path provided'}), 400

            self.save(data['file_path'])
            return jsonify({'message': 'Model saved successfully'})

        @self.app.route('/load_model', methods=['POST'])
        def load_model():
            data = request.json
            if 'file_path' not in data:
                return jsonify({'error': 'No file path provided'}), 400

            self.load(data['file_path'])
            return jsonify({'message': 'Model loaded successfully'})

        @self.app.route('/get_model_info', methods=['GET'])
        def get_model_info():
            return jsonify({
                'num_layers': len(self.layers),
                'temperature': self.temperature,
                'debug_mode': self.debug_mode
            })

    def _is_safe_size(self, size):
        # Check if the size is safe and won't cause overflow or NaN
        return size <= 512

    def suggest_architecture(self, input_size, output_size, task_type='classification', data_type='tabular', depth=3,
                             temperature=1.0):
        self._check_breakpoint('suggest_architecture', input_size, output_size, task_type, data_type, depth,
                               temperature)

        suggested_architecture = []
        current_size = min(input_size, 1024)  # Cap initial size

        # Define layer groups and activation layers
        layer_groups = {
            'image': [Conv2DLayer, AveragePoolingLayer],
            'sequence': [LSTMLayer, GRULayer, RNNLayer, ScaledDotProductAttention],
            'tabular': [DenseLayer]
        }
        activation_layers = [ELUActivationLayer, SoftsignActivationLayer, HardTanhActivationLayer, SELUActivationLayer,
                             LinearActivationLayer, PReLUActivationLayer, TanhActivationLayer, SwishActivationLayer, GELUActivationLayer]
        regularization_layers = [L1RegularizationLayer, L2RegularizationLayer, L3RegularizationLayer]
        conv_layers = [Conv1DLayer, Conv2DLayer, Conv3DLayer]
        attention_layers = [ScaledDotProductAttention, MultiHeadAttention]
        padding_layers = [ZeroPaddingLayer, ReflectionPaddingLayer, PaddingLayer]
        fold_unfold_layers = [Fold, Unfold]

        main_layers = layer_groups.get(data_type, layer_groups['tabular'])

        # Input layer
        if data_type == 'image':
            suggested_architecture.append(
                f"network.add(Conv2DLayer(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1))")
            current_size = 32
        elif data_type == 'sequence':
            hidden_size = min(max(32, current_size), 512)
            suggested_architecture.append(
                f"network.add(LSTMLayer(input_size={current_size}, hidden_size={hidden_size}))")
            current_size = hidden_size
        else:
            hidden_size = min(max(32, current_size), 512)
            suggested_architecture.append(f"network.add(DenseLayer({current_size}, {hidden_size}))")
            current_size = hidden_size

        # Hidden layers
        for i in range(depth - 1):
            try:
                layer = random.choice(main_layers)
                if layer in conv_layers:
                    if data_type == 'tabular':
                        continue  # Skip conv layers for tabular data
                    next_size = min(current_size * 2, 512)
                    if not self._is_safe_size(next_size):
                        continue  # Skip if the next size is not safe
                    suggested_architecture.append(
                        f"network.add({layer.__name__}(in_channels={current_size}, out_channels={next_size}, kernel_size=3, stride=1, padding=1))")
                    if i % 2 == 1:
                        pool_layer = random.choice([MaxPoolingLayer, AveragePoolingLayer])
                        suggested_architecture.append(f"network.add({pool_layer.__name__}(pool_size=2, stride=2))")
                elif layer in (LSTMLayer, GRULayer, RNNLayer):
                    next_size = min(current_size * 2, 512)
                    if not self._is_safe_size(next_size):
                        continue  # Skip if the next size is not safe
                    suggested_architecture.append(
                        f"network.add({layer.__name__}(input_size={current_size}, hidden_size={next_size}))")
                elif layer in padding_layers:
                    padding_size = (1, 1) if isinstance(layer, ReflectionPaddingLayer) else 1
                    suggested_architecture.append(f"network.add({layer.__name__}(padding={padding_size}))")
                    next_size = current_size
                elif layer in attention_layers:
                    suggested_architecture.append(f"network.add({layer.__name__}(d_model={current_size}))")
                    next_size = current_size
                elif layer in fold_unfold_layers:
                    suggested_architecture.append(f"network.add({layer.__name__}(kernel_size=3, stride=1, padding=1))")
                    next_size = current_size
                elif layer == PairwiseDistance:
                    suggested_architecture.append(f"network.add({layer.__name__}(p=2, eps=1e-6, keepdim=True))")
                    next_size = current_size
                elif layer == Embedding:
                    suggested_architecture.append(
                        f"network.add({layer.__name__}(vocab_size=5000, embedding_dim={current_size}))")
                    next_size = current_size
                else:
                    next_size = min(current_size * 2, 512)
                    if not self._is_safe_size(next_size):
                        continue  # Skip if the next size is not safe
                    suggested_architecture.append(f"network.add({layer.__name__}({current_size}, {next_size}))")

                current_size = next_size

                activation = self._choose_activation(activation_layers, temperature)
                suggested_architecture.append(self._format_activation(activation))
                regularization_applied = False

                if random.random() < 0.3:
                    norm_layer = BatchNormLayer
                    suggested_architecture.append(f"network.add({norm_layer.__name__}(num_features={current_size}))")

                    if not regularization_applied and random.random() < 0.5:
                        reg_layer = random.choice(regularization_layers)
                        suggested_architecture.append(
                            f"network.add({reg_layer.__name__}(layer=network.layers[-1], lambda_=0.01))")
                        regularization_applied = True

                if random.random() < 0.2:
                    suggested_architecture.append(f"network.add(DropoutLayer())")

            except Exception as e:
                print(f"Error adding layer: {e}")
                self.error_history.append(str(e))
                continue

        if data_type != 'tabular':
            suggested_architecture.append(f"network.add(Flatten())")
        suggested_architecture.append(f"network.add(DenseLayer({current_size}, {output_size}))")

        suggested_architecture.append(self._get_final_activation(task_type))

        if task_type == 'classification' or data_type == 'image':
            optimizer = "AdamOptimizer(lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8)"
        elif task_type == 'regression' or data_type == 'sequence':
            optimizer = "NadamOptimizer(epsilon=1e-8)"
        else:
            optimizer = "AdadeltaOptimizer(epsilon=1e-8)"

        return suggested_architecture, optimizer

    def _choose_activation(self, activation_layers, temperature):
        if temperature < 0.5:
            return ReLUActivationLayer
        elif temperature > 1.5:
            return LeakyReLUActivationLayer
        else:
            return random.choice(activation_layers)

    def _format_activation(self, activation):
        if activation in [LeakyReLUActivationLayer, SELUActivationLayer]:
            return f"network.add({activation.__name__}(alpha=0.1))"
        else:
            return f"network.add({activation.__name__}())"

    def _get_final_activation(self, task_type):
        if task_type == 'classification':
            return "network.add(SoftmaxActivationLayer())"
        elif task_type == 'regression':
            return "network.add(LinearActivationLayer())"
        else:
            raise ValueError(f"Unsupported task type: {task_type}")

    def apply_suggested_architecture(self, input_size, output_size, task_type='classification', data_type='tabular',
                                     depth=3):
        suggested_architecture, suggested_optimizer = self.suggest_architecture(input_size, output_size, task_type,
                                                                                data_type, depth)
        for layer_str in suggested_architecture:
            exec(layer_str, globals(), {'network': self})

        return suggested_architecture, suggested_optimizer

    def train_with_suggested_architecture(self, inputs, targets, input_size, output_size, optimizer=None,
                                          task_type='classification', data_type='tabular', depth=3, epochs=100,
                                          lr=0.01, batch_size=32):
        self._check_breakpoint('train_with_suggested_architecture', inputs, targets, input_size, output_size,
                               task_type, data_type, depth, epochs, lr, batch_size)

        suggested_architecture, suggested_optimizer = self.apply_suggested_architecture(input_size, output_size,
                                                                                        task_type, data_type, depth)
        # Print the suggested architecture
        print("Applied Architecture:")
        for layer in suggested_architecture:
            print(layer)

        # Print the suggested optimizer
        print(f"Suggested Optimizer: {suggested_optimizer}")

        # Use suggested optimizer if none is provided
        if optimizer is None:
            optimizer = suggested_optimizer

        # Prepare inputs and targets
        if not isinstance(inputs, Tensor):
            inputs = Tensor(np.array(inputs))
        if not isinstance(targets, Tensor):
            targets = Tensor(np.array(targets))

        # Choose appropriate loss function
        if task_type == 'classification':
            loss_function = CrossEntropyLoss()
        elif task_type == 'image':
            loss_function = MeanAbsoluteError()
        else:  # regression
            loss_function = MSELoss()

        # Train the network
        losses = self.train(inputs, targets, epochs, lr, optimizer, batch_size, loss_function)

        # Plot the training loss
        self.plot_loss(losses)

        return losses

    def rank_network_performance(self, test_inputs, test_targets, temperature, task_type='classification',
                                 creativity_threshold=0.5):
        self._check_breakpoint('rank_network_performance', test_inputs, test_targets, temperature, task_type,
                               creativity_threshold)

        # Start timing
        start_time = time.time()

        # Start memory tracking
        process = psutil.Process()
        start_memory = process.memory_info().rss

        # Set the network's temperature
        self.set_temperature(temperature)

        # Forward pass
        outputs = self.forward(Tensor(test_inputs))

        # Calculate loss
        if task_type == 'classification':
            loss_function = CrossEntropyLoss()
        else:  # regression
            loss_function = MSELoss()

        loss = loss_function.forward(outputs, Tensor(test_targets))

        # Calculate accuracy
        if task_type == 'classification':
            predictions = np.argmax(outputs.data, axis=1)
            accuracy = np.mean(predictions == np.argmax(test_targets, axis=1))
        else:
            accuracy = 1.0 - np.mean(np.abs(outputs.data - test_targets) / test_targets)

        # Calculate output diversity (as a simple measure of creativity)
        output_diversity = np.std(outputs.data)

        # Determine expected creativity level based on temperature
        expected_creativity = 1.0 if temperature > 1.0 else 0.0 if temperature < 1.0 else 0.5

        # Calculate creativity alignment score
        creativity_alignment = 1.0 - abs(output_diversity - expected_creativity)

        # Calculate overall performance score
        performance_score = (accuracy * (1 - creativity_threshold) +
                             creativity_alignment * creativity_threshold)

        # Rank the performance
        if performance_score > 0.8:
            rank = "Excellent"
        elif performance_score > 0.6:
            rank = "Good"
        elif performance_score > 0.4:
            rank = "Fair"
        else:
            rank = "Poor"

        # Stop timing and memory tracking
        end_time = time.time()
        end_memory = process.memory_info().rss

        # Calculate elapsed time and memory usage
        elapsed_time = end_time - start_time
        memory_usage = (end_memory - start_memory) / (1024 ** 2)  # Convert bytes to MB

        result = {
            "temperature": temperature,
            "loss": float(np.mean(loss.data)),
            "accuracy": float(accuracy),
            "output_diversity": float(output_diversity),
            "creativity_alignment": float(creativity_alignment),
            "performance_score": float(performance_score),
            "rank": rank,
            "execution_time_seconds": elapsed_time,
            "memory_usage_mb": memory_usage
        }

        self.logger.info(f"Performance Ranking: {result}")

        return result
