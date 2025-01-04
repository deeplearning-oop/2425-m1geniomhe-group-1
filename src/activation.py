from abc import ABC, abstractmethod
from module import Module
from tensor import Tensor
import numpy as np

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


class ReLU(Activation):
    """
    ReLU (Rectified Linear Unit) activation function.
    Applies max(0, x) element-wise to the input tensor.
    """
    def forward(self, x):
        out_data = np.maximum(x.data, 0) #apply ReLU operation (max(0, x)) element-wise to input data
        out = Tensor(out_data, requires_grad=x.requires_grad) #create a new tensor to store the result
        out.parents = {x}

        if x.requires_grad:  #if the input tensor requires gradients, define the backward pass
            def _backward(grad):
                relu_grad = (x.data > 0).astype(float)  #gradient for ReLU: 1 for positive values, 0 for negative
                if x.grad is None:
                    x.grad = grad * relu_grad
                else:
                    x.grad += grad * relu_grad
            out.grad_fn = _backward
            out.grad_fn_name = "ReLUBackward"
        return out
    
    def __repr__(self):
        return "ReLU()"


class Softmax(Activation):
    """
    Softmax activation function.
    Applies the softmax operation to the input tensor, typically used for classification tasks.
    Converts logits into probabilities.
    """
    def forward(self, x):
        #Numerically stable softmax: subtracting max before exponentiation
        max_logits = np.max(x.data, axis=0, keepdims=True)  #max value for stability
        exps = np.exp(x.data - max_logits)  #exponentiate input
        sum_exps = np.sum(exps, axis=0, keepdims=True)  #sum across the batch
        result = exps / sum_exps  #softmax output (probabilities)

        out = Tensor(result, requires_grad=x.requires_grad) #create a new tensor to store the softmax output
        out.parents = {x}

        if x.requires_grad: #if the input tensor requires gradients, define the backward pass
            def _backward(grad):
                grad_input = result * (grad - np.sum(grad * result, axis=0, keepdims=True))
                if x.grad is None:
                    x.grad = grad_input
                else:
                    x.grad += grad_input
            out.grad_fn = _backward
            out.grad_fn_name = "SoftmaxBackward"
        return out
    
    def __repr__(self):
        return "Softmax()"


