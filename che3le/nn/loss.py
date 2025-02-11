__doc__ = '''
-------------------
Loss module
-------------------

This module contains the Loss class which is used to compute the loss between the target labels and the predicted values
'''

__all__ = ["Loss", "MSE", "CrossEntropyLoss", "BinaryCrossEntropyLoss", "NegativeLogLikelihoodLoss"]

from abc import ABC, abstractmethod
from che3le.nn.module import Module
from che3le.tensor import Tensor
import numpy as np

class Loss(Module):
    def __init__(self):
        super().__init__()
        self._cache = None

    @abstractmethod
    def forward(self, y, y_hat):
        raise NotImplementedError

    def __call__(self, y, y_hat):
        return self.forward(y, y_hat)
    
    @staticmethod
    def backward_decorator(loss_type):
        def decorator(func):
            def wrapper(self, y, y_hat):
                # Perform the forward pass to create the output
                out = func(self, y, y_hat)
                # Attach the grad function to the result tensor
                out.grad_fn = lambda grad: self.grad_compute(loss_type, y, y_hat)
                out.grad_fn_name = f"{loss_type}Backward"
                # Define the parents (same as in the Tensor class)
                out.parents = {y, y_hat}
                return out
            return wrapper
        return decorator

    def grad_compute(self, loss_type, y, y_hat):
        """
        Centralized gradient computation for different loss functions.
        """
        if loss_type == "MSE":
            # Gradient of MSE loss with respect to y_hat
            batch_size = y_hat.data.shape[0]
            grad_input = 2 * (y_hat.data - y.data) / batch_size

        elif loss_type == "CrossEntropyLoss":
            
            # One-hot encoding of y
            one_hot_y = np.zeros((y.data.size, y_hat.data.shape[0]))
            one_hot_y[np.arange(y.data.size), y.data] = 1
            one_hot_y = one_hot_y.T
            
            grad_input = - (one_hot_y / y_hat.data) / y.data.size
            
        elif loss_type == "BinaryCrossEntropyLoss":
            # Gradient of Binary Cross-Entropy Loss with respect to y_hat
            grad_input = -(y.data / y_hat.data) + (1 - y.data) / (1 - y_hat.data)
            grad_input /= y.data.size
        
        elif loss_type == "NegativeLogLikelihoodLoss":
            # Adding epsilon to avoid division by zero
            epsilon = 1e-15
            y_pred = np.clip(y_hat.data, epsilon, 1 - epsilon)

            batch_size = y.data.size
            grad_input = np.zeros_like(y_hat.data)

            # For each sample, set the gradient of the true class to -1 / p(y_true)
            grad_input[np.arange(batch_size), y.data] = -1 / y_pred[np.arange(batch_size), y.data]
            grad_input /= batch_size
        
        y_hat.grad = grad_input if y_hat.grad is None else y_hat.grad + grad_input

        # Add more loss functions here if needed
        return  y_hat.grad
         

class MSE(Loss):
    @Loss.backward_decorator("MSE")
    def forward(self, y, y_hat):
        """
        y: Tensor of shape (batch_size, num_outputs) (target labels)
        y_hat: Tensor of shape (batch_size, num_outputs) (predicted values)
        """
        # Compute Mean Squared Error
        error = y_hat.data - y.data
        loss = np.mean(error ** 2)
        return Tensor(loss, requires_grad=True, is_leaf=False)

    def __repr__(self):
        return "MSE()"
    

class CrossEntropyLoss(Loss):
    @Loss.backward_decorator("CrossEntropyLoss")
    def forward(self, y, y_hat):
        """
        y: Tensor of shape (batch_size, num_outputs) (target labels)
        y_hat: Tensor of shape (batch_size, num_outputs) (predicted values)
        """
        # Adding epsilon to avoid log(0)
        epsilon = 1e-15
        y_pred = np.clip(y_hat.data, epsilon, 1 - epsilon)

        # One-hot encoding y
        one_hot_y = np.zeros((y.data.size, y_hat.data.shape[0]))
        one_hot_y[np.arange(y.data.size), y.data] = 1
        one_hot_y = one_hot_y.T

        loss = -np.sum(one_hot_y * np.log(y_pred)) / y.data.size
        return Tensor(loss, requires_grad=True, is_leaf=False)
    
    def __repr__(self):
        return "CrossEntropyLoss()"
        
class BinaryCrossEntropyLoss(Loss):
    @Loss.backward_decorator("BinaryCrossEntropyLoss")
    def forward(self, y, y_hat):
        """
        y: Ground truth tensor (batch_size,)
        y_hat: Predicted probabilities tensor (batch_size,). Sigmoid output.
        """
        # Adding epsilon to avoid log(0)
        epsilon = 1e-15
        y_hat = np.clip(y_hat.data, epsilon, 1 - epsilon)

        # Binary Cross-Entropy Loss
        loss = -np.mean(y.data * np.log(y_hat) + (1 - y.data) * np.log(1 - y_hat))

        return Tensor(loss, requires_grad=True, is_leaf=False)

    def __repr__(self):
        return "BinaryCrossEntropyLoss()"
    
class NegativeLogLikelihoodLoss(Loss):
    @Loss.backward_decorator("NegativeLogLikelihoodLoss")
    def forward(self, y, y_hat):
        """
        y: Tensor of shape (batch_size,) (target labels - integer class labels)
        y_hat: Tensor of shape (batch_size, num_classes) (predicted class probabilities)
        """
        # Adding epsilon to avoid log(0)
        epsilon = 1e-15
        y_pred = np.clip(y_hat.data, epsilon, 1 - epsilon)

        # The true labels are in integer form, so we use the correct class probabilities
        batch_size = y.data.size
        correct_class_probs = y_pred[np.arange(batch_size), y.data]

        # Negative Log-Likelihood loss
        loss = -np.sum(np.log(correct_class_probs)) / batch_size
        
        return Tensor(loss, requires_grad=True, is_leaf=False)

    def __repr__(self):
        return "NegativeLogLikelihoodLoss()"

