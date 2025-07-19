import numpy as np
from typing import List
from .base import Module


class Linear(Module):
    """
    Applies linear (affine) transformation of data: y = x W^T + b
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        """
        :param in_features: input vector features
        :param out_features: output vector features
        :param bias: whether to use additive bias
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = np.random.uniform(-1, 1, (out_features, in_features)) / np.sqrt(in_features)
        self.bias = np.random.uniform(-1, 1, out_features) / np.sqrt(in_features) if bias else None

        self.grad_weight = np.zeros_like(self.weight)
        self.grad_bias = np.zeros_like(self.bias) if bias else None

    def compute_output(self, input: np.ndarray) -> np.ndarray:
        """
        :param input: array of shape (batch_size, in_features)
        :return: array of shape (batch_size, out_features)
        """
        if self.bias is not None:
            output = input @ self.weight.T + self.bias
        else:
            output = input @ self.weight.T
        return output

    def compute_grad_input(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        """
        :param input: array of shape (batch_size, in_features)
        :param grad_output: array of shape (batch_size, out_features)
        :return: array of shape (batch_size, in_features)
        """
        grad_input = grad_output @ self.weight
        return grad_input

    def update_grad_parameters(self, input: np.ndarray, grad_output: np.ndarray):
        """
        :param input: array of shape (batch_size, in_features)
        :param grad_output: array of shape (batch_size, out_features)
        """
        self.grad_weight += grad_output.T @ input
        if self.bias is not None:
            self.grad_bias += np.sum(grad_output, axis=0)

    def zero_grad(self):
        self.grad_weight.fill(0)
        if self.bias is not None:
            self.grad_bias.fill(0)

    def parameters(self) -> List[np.ndarray]:
        if self.bias is not None:
            return [self.weight, self.bias]

        return [self.weight]

    def parameters_grad(self) -> List[np.ndarray]:
        if self.bias is not None:
            return [self.grad_weight, self.grad_bias]

        return [self.grad_weight]

    def __repr__(self) -> str:
        out_features, in_features = self.weight.shape
        return f'Linear(in_features={in_features}, out_features={out_features}, ' \
               f'bias={not self.bias is None})'


class BatchNormalization(Module):
    """
    Applies batch normalization transformation
    """
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1, affine: bool = True):
        """
        :param num_features:
        :param eps:
        :param momentum:
        :param affine: whether to use trainable affine parameters
        """
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.affine = affine

        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)

        self.weight = np.ones(num_features) if affine else None
        self.bias = np.zeros(num_features) if affine else None

        self.grad_weight = np.zeros_like(self.weight) if affine else None
        self.grad_bias = np.zeros_like(self.bias) if affine else None

        # store this values during forward path and re-use during backward pass
        self.mean = None
        self.input_mean = None  # input - mean
        self.var = None
        self.sqrt_var = None
        self.inv_sqrt_var = None
        self.norm_input = None

    def compute_output(self, input: np.ndarray) -> np.ndarray:
        """
        :param input: array of shape (batch_size, num_features)
        :return: array of shape (batch_size, num_features)
        """
        if self.training:
            self.mean = np.mean(input, axis=0)
            self.input_mean = input - self.mean[np.newaxis, :]
            self.var = np.mean(self.input_mean ** 2, axis=0)
            self.sqrt_var = np.sqrt(self.var + self.eps)
            self.inv_sqrt_var = 1 / self.sqrt_var

            output = self.input_mean * self.inv_sqrt_var[np.newaxis, :]
            
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * self.mean
            B = input.shape[0]
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * B * self.var / (B - 1)

        else:
            num = input - self.running_mean[np.newaxis, :]
            den = np.sqrt(self.running_var + self.eps)
            output = num / den[np.newaxis, :]

        if self.affine:
            output = output * self.weight[np.newaxis, :] + self.bias[np.newaxis, :]
            
        return output

    def compute_grad_input(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        """
        :param input: array of shape (batch_size, num_features)
        :param grad_output: array of shape (batch_size, num_features)
        :return: array of shape (batch_size, num_features)
        """
        if self.training:
            B = input.shape[0]
            input_mean = input - self.mean[np.newaxis, :]
            term1 = grad_output
            term2 = np.mean(grad_output, axis=0, keepdims=True)
            term3 = input_mean * np.sum(input_mean * grad_output, axis=0)[np.newaxis, :] / (B * (self.var + self.eps)[np.newaxis, :])
            grad_input = (term1 - term2 - term3) * self.inv_sqrt_var[np.newaxis, :]
        
        else:
            grad_input = grad_output / np.sqrt(self.running_var + self.eps)[np.newaxis, :]
        
        if self.affine:
            grad_input = grad_input * self.weight[np.newaxis, :]

        return grad_input

    def update_grad_parameters(self, input: np.ndarray, grad_output: np.ndarray):
        """
        :param input: array of shape (batch_size, num_features)
        :param grad_output: array of shape (batch_size, num_features)
        """
        if self.affine:
            if self.training:
                num = input - self.mean[np.newaxis, :]
                X = num * self.inv_sqrt_var[np.newaxis, :]
            else:
                num = input - self.running_mean[np.newaxis, :]
                den = np.sqrt(self.running_var + self.eps)
                X = num / den[np.newaxis, :]

            self.grad_weight += np.sum(X * grad_output, axis=0)
            self.grad_bias += np.sum(grad_output, axis=0)


    def zero_grad(self):
        if self.affine:
            self.grad_weight.fill(0)
            self.grad_bias.fill(0)

    def parameters(self) -> List[np.ndarray]:
        return [self.weight, self.bias] if self.affine else []

    def parameters_grad(self) -> List[np.ndarray]:
        return [self.grad_weight, self.grad_bias] if self.affine else []

    def __repr__(self) -> str:
        return f'BatchNormalization(num_features={len(self.running_mean)}, ' \
               f'eps={self.eps}, momentum={self.momentum}, affine={self.affine})'


class Dropout(Module):
    """
    Applies dropout transformation
    """
    def __init__(self, p=0.5):
        super().__init__()
        assert 0 <= p < 1
        self.p = p
        self.mask = None

    def compute_output(self, input: np.ndarray) -> np.ndarray:
        """
        :param input: array of an arbitrary size
        :return: array of the same size
        """
        if self.training:
            self.mask = np.random.binomial(n=1, p=1 - self.p, size=input.shape)
            output = self.mask * input / (1 - self.p)
        else:
            output = input
        return output

    def compute_grad_input(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        """
        :param input: array of an arbitrary size
        :param grad_output: array of the same size
        :return: array of the same size
        """
        if self.training:
            grad_input = self.mask * grad_output / (1 - self.p)
        else:
            grad_input = grad_output
        return grad_input

    def __repr__(self) -> str:
        return f'Dropout(p={self.p})'


class Sequential(Module):
    """
    Container for consecutive application of modules
    """
    def __init__(self, *args):
        super().__init__()
        self.modules = list(args)
        self.outputs = None

    def compute_output(self, input: np.ndarray) -> np.ndarray:
        """
        :param input: array of size matching the input size of the first layer
        :return: array of size matching the output size of the last layer
        """
        self.outputs = [input]
        n = len(self.modules)
        for i in range(n):
            self.outputs.append(self.modules[i].compute_output(self.outputs[i]))
        return self.outputs[-1]

    def compute_grad_input(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        """
        :param input: array of size matching the input size of the first layer
        :param grad_output: array of size matching the output size of the last layer
        :return: array of size matching the input size of the first layer
        """
        grad_input = grad_output
        n = len(self.modules)
        for i in range(n):
            grad_input = self.modules[n - 1 - i].compute_grad_input(self.outputs[n - 1 - i], grad_input)
        return grad_input
    
    def update_grad_parameters(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        grad_input = grad_output
        n = len(self.modules)
        for i in range(n):
            self.modules[n - 1 - i].update_grad_parameters(self.outputs[n - 1 - i], grad_input)
            grad_input = self.modules[n - 1 - i].compute_grad_input(self.outputs[n - 1 - i], grad_input)

    def __getitem__(self, item):
        return self.modules[item]

    def train(self):
        for module in self.modules:
            module.train()

    def eval(self):
        for module in self.modules:
            module.eval()

    def zero_grad(self):
        for module in self.modules:
            module.zero_grad()

    def parameters(self) -> List[np.ndarray]:
        return [parameter for module in self.modules for parameter in module.parameters()]

    def parameters_grad(self) -> List[np.ndarray]:
        return [grad for module in self.modules for grad in module.parameters_grad()]

    def __repr__(self) -> str:
        repr_str = 'Sequential(\n'
        for module in self.modules:
            repr_str += ' ' * 4 + repr(module) + '\n'
        repr_str += ')'
        return repr_str
