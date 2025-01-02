from module import Module
from tensor import Tensor
import numpy as np

class Loss(Module):
    def __init__(self):
        super().__init__()
        self._cache = None
    def __call__(self, y, y_hat):
        return self.forward(y, y_hat)

class MSE(Loss):
    def forward(self, y, y_hat):
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
    def forward(self, y, y_hat):
        
        epsilon = 1e-15
        y_pred = np.clip(y_hat.data, epsilon, 1. - epsilon)  # Clip values to avoid log(0)
        y_one_hot = np.eye(y_pred.shape[1])[y.data]
        # Calculate the cross-entropy loss
        loss = -np.sum(y_one_hot * np.log(y_pred)) / y.data.shape[0]
        
        out = Tensor(loss, requires_grad=True)
        
        # Keep track of the parents (tensors used in this computation)
        out.parents = {y, y_hat}
        
        # Backward function to compute gradients
        def _backward(grad):
            
            grad_input = y_pred - y_one_hot

            if y_hat.grad is None:
                y_hat.grad = grad_input
            else:
                y_hat.grad += grad_input

        out.grad_fn = _backward
        out.grad_fn_name = "CrossEntropyLossBackward"
        
        return out


    # def backward(self):
    #     y, y_hat = self._cache
    #     y_one_hot = Tensor(np.eye(y_hat.shape[1])[y.data], requires_grad=False)
    #     return -y_one_hot / (y_hat + Tensor(1e-9, requires_grad=False))