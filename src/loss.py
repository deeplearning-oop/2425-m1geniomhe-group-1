from module import Module
from tensor import Tensor
import numpy as np

class Loss(Module):
    def __init__(self):
        super().__init__()
        self._cache = None
    def loss_fn(self, y, y_hat):
        raise NotImplementedError
    def __call__(self, y, y_hat):
        return self.loss_fn(y, y_hat)
    def forward(self, y, y_hat): #implementing the forward method because it inherits the abstract class Module
        return self.loss_fn(y, y_hat)

class MSE(Loss):
    def loss_fn(self, y, y_hat):
        """
        y: Tensor of shape (batch_size, num_outputs) (target labels)
        y_hat: Tensor of shape (batch_size, num_outputs) (predicted values)
        """
        # Compute Mean Squared Error
        batch_size = y_hat.data.shape[0]
        error = y_hat.data - y.data
        loss = np.mean(error ** 2)
        
        # Create the output tensor with requires_grad=True for gradient computation
        out = Tensor(loss, requires_grad=True)
        
        # Keep track of the parents (tensors used in this computation)
        out.parents = {y, y_hat}
        
        # Backward function to compute gradients
        def _backward(grad):
            # Gradient of the loss w.r.t y_hat
            
            grad_input = 2 * (y_hat.data - y.data) / batch_size
            
            if y_hat.grad is None:
                y_hat.grad = grad_input
            else:
                y_hat.grad += grad_input

        out.grad_fn = _backward
        out.grad_fn_name = "MSEBackward"
        
        return out
    

import numpy as np

class CrossEntropyLoss(Loss):
    def loss_fn(self, y, y_hat):
        
        epsilon = 1e-15
        y_pred = np.clip(y_hat.data, epsilon, 1 - epsilon)
        
        one_hot_y = np.zeros((y.data.size, y_hat.data.shape[0]))
        one_hot_y[np.arange(y.data.size), y.data] = 1
        one_hot_y = one_hot_y.T
        
        loss = -np.sum(one_hot_y * np.log(y_pred)) / y.data.size
        
        out = Tensor(loss, requires_grad=True)
        
        # Keep track of the parents (tensors used in this computation)
        out.parents = {y, y_hat}
        
        # Backward function to compute gradients
        def _backward(grad):
                        
            # grad_input = (y_hat.data - one_hot_y) / y.data.size
            grad_input = - (one_hot_y / y_pred) / y.data.size
            
            if y_hat.grad is None:
                y_hat.grad = grad_input
            else:
                y_hat.grad += grad_input

        out.grad_fn = _backward
        out.grad_fn_name = "CrossEntropyLossBackward"
        
        return out

