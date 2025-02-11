__doc__='''
-----------------------
Activation module
-----------------------
This module contains the activation functions used in neural networks
Including: ReLU, Softmax, Sigmoid
'''

__all__ = ["Activation", "ReLU", "Softmax", "Sigmoid"]

from abc import ABC, abstractmethod
import numpy as np
from che3le.nn.module import Module
from che3le.tensor import Tensor

class Activation(Module):
    """
    Base class for the activation functions.
    All activation functions should inherit from this class and implement the 'forward' method.
    """
    @abstractmethod
    def forward(self, x):
        """
        Abstract forward pass method must be implemented by any subclass.
        Each activation function applies its operation to the input tensor.
        """
        pass
    
    def __call__(self, x):
        return self.forward(x)
    
    @staticmethod
    def backward_decorator(op_type):
        def decorator(func):
            def wrapper(self, x):
                # Perform the forward pass and create the result tensor
                out = func(self, x)
                # Attach the grad function to the result tensor
                out.grad_fn = lambda grad: self.grad_compute(grad, op_type, x, out)
                out.grad_fn_name = f"{op_type}Backward"
                # Define the parents
                out.parents = {x} 
                return out
            return wrapper
        return decorator
    
    def grad_compute(self, grad, op_type, x, out):
        """
        Centralizes gradient computation for different activation functions.
        """
        if op_type == "ReLU":
            # Gradient computation for ReLU: 1 for positive values, 0 for negative
            relu_grad = (x.data > 0).astype(float)
            x.grad = grad * relu_grad if x.grad is None else x.grad + grad * relu_grad
            
        elif op_type == "Softmax":
            
            softmax_grad = out.data * (grad - np.sum(grad * out.data, axis=0, keepdims=True))
            x.grad = softmax_grad if x.grad is None else x.grad + softmax_grad
        
        elif op_type == "Sigmoid":
            # Sigmoid gradient computation
            sigmoid_grad = out.data * (1 - out.data)  # sigmoid' = sigmoid * (1 - sigmoid)
            x.grad = grad * sigmoid_grad if x.grad is None else x.grad + grad * sigmoid_grad
        
        
        return x.grad  # Return the updated gradient


class ReLU(Activation):
    """
    ReLU (Rectified Linear Unit) activation function.
    Applies max(0, x) element-wise to the input tensor.
    """
    @Activation.backward_decorator("ReLU")
    def forward(self, x):
        return Tensor(np.maximum(0, x.data), requires_grad=x.requires_grad, is_leaf=False)
    
    def __repr__(self):
        return "ReLU()"


class Softmax(Activation):
    """
    Softmax activation function.
    Applies the softmax operation to the input tensor, typically used for classification tasks.
    Converts logits into probabilities.
    """
    @Activation.backward_decorator("Softmax")
    def forward(self, x):
        # Numerically stable softmax: subtracting max before exponentiation
        max_logits = np.max(x.data, axis=0, keepdims=True)  # max value for stability
        exps = np.exp(x.data - max_logits)  # exponentiate input
        sum_exps = np.sum(exps, axis=0, keepdims=True)  # sum across the batch
        result = exps / sum_exps  # softmax output (probabilities)

        out = Tensor(result, requires_grad=x.requires_grad, is_leaf=False)  # create a new tensor for the softmax result
        return out
    
    def __repr__(self):
        return "Softmax()"
    
class Sigmoid(Activation):
    """
    Sigmoid activation function.
    Applies the sigmoid function element-wise to the input tensor.
    """
    @Activation.backward_decorator("Sigmoid")
    def forward(self, x):
        # Sigmoid function
        result = 1 / (1 + np.exp(-x.data))  # Sigmoid activation
        out = Tensor(result, requires_grad=x.requires_grad, is_leaf=False)  # create a new tensor for the sigmoid result
        return out
    
    def __repr__(self):
        return "Sigmoid()"


