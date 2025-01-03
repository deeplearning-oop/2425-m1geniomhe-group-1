import numpy as np

class Tensor:
    def __init__(self, data, requires_grad=False):
        self.data = np.array(data) if not isinstance(data, np.ndarray) else data
        self.requires_grad = requires_grad
        self.grad = None
        self.grad_fn = None
        self.grad_fn_name = None
        self.parents = set()

    @property
    def shape(self):
        return self.data.shape
    
    def __getitem__(self, idx):
        return self.data[idx]

    def __add__(self, other):
        
        other = other if isinstance(other, Tensor) else Tensor(other)
        result = Tensor(self.data + other.data, requires_grad=self.requires_grad or other.requires_grad)
        result.parents = {self, other}

        def _backward(grad):
            if self.requires_grad:
                if self.grad is None:
                    self.grad = grad
                else:
                    self.grad += grad
            if other.requires_grad:
                if other.grad is None:
                    other.grad = grad
                else:
                    other.grad += grad

        result.grad_fn = _backward
        result.grad_fn_name = "AddBackward"
        return result

    def __neg__(self):
        
        result = Tensor(-self.data, requires_grad=self.requires_grad)
        result.parents = {self}

        def _backward(grad):
            if self.requires_grad:
                if self.grad is None:
                    self.grad = -grad
                else:
                    self.grad -= grad

        result.grad_fn = _backward
        result.grad_fn_name = "NegBackward"
        return result

    def __sub__(self, other):
        
        other = other if isinstance(other, Tensor) else Tensor(other)
        result = Tensor(self.data - other.data, requires_grad=self.requires_grad or other.requires_grad)
        result.parents = {self, other}
        
        def _backward(grad):
            if self.requires_grad:
                if self.grad is None:
                    self.grad = grad
                else:
                    self.grad += grad
            if other.requires_grad:
                if other.grad is None:
                    other.grad = -grad
                else:
                    other.grad -= grad
        
        result.grad_fn = _backward
        result.grad_fn_name = "SubBackward"
        return result

    def __mul__(self, other):
        # Handle the case when 'other' is a scalar (e.g., a float or int)
        if isinstance(other, (int, float)) or isinstance(self, (int, float)):
            # Scalar multiplication: Multiply the scalar with the data and return a new Tensor
            out = Tensor(self.data * other, requires_grad=self.requires_grad)
            out.parents = {self}

            def _backward(grad):
                if self.requires_grad:
                    if self.grad is None:
                        self.grad = grad * other  # Gradient w.r.t. the scalar
                    else:
                        self.grad += grad * other  # Accumulate gradient w.r.t. the scalar

            out.grad_fn = _backward
            out.grad_fn_name = "ScalarMulBackward"
            return out
        
        # Handle the case when 'other' is a Tensor
        if isinstance(other, Tensor):
            out = Tensor(self.data * other.data, requires_grad=self.requires_grad or other.requires_grad)
            out.parents = {self, other}

            def _backward(grad):
                if self.requires_grad:
                    if self.grad is None:
                        self.grad = grad * other.data  # Gradient w.r.t. the other Tensor
                    else:
                        self.grad += grad * other.data  # Accumulate gradient w.r.t. the other Tensor
                if other.requires_grad:
                    if other.grad is None:
                        other.grad = grad * self.data  # Gradient w.r.t. self Tensor
                    else:
                        other.grad += grad * self.data  # Accumulate gradient w.r.t. self Tensor

            out.grad_fn = _backward
            out.grad_fn_name = "TensorMulBackward"
            return out


    def __truediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data / other.data, requires_grad=self.requires_grad or other.requires_grad)
        out.parents = {self, other}

        def _backward(grad):
            if self.requires_grad:
                if self.grad is None:
                    self.grad = grad / other.data
                else:
                    self.grad += grad / other.data
            if other.requires_grad:
                if other.grad is None:
                    other.grad = -grad * self.data / (other.data ** 2)
                else:
                    other.grad -= grad * self.data / (other.data ** 2)

        out.grad_fn = _backward
        out.grad_fn_name = "DivBackward"
        return out

    def mean(self):
        out = Tensor(self.data.mean(), requires_grad=self.requires_grad)
        out.parents = {self}

        def _backward(grad):
            if self.requires_grad:
                if self.grad is None:
                    self.grad = grad / self.data.size
                else:
                    self.grad += grad / self.data.size

        out.grad_fn = _backward
        out.grad_fn_name = "MeanBackward"
        return out

    def sum(self):
        out = Tensor(self.data.sum(), requires_grad=self.requires_grad)
        out.parents = {self}

        def _backward(grad):
            if self.requires_grad:
                if self.grad is None:
                    self.grad = grad * np.ones_like(self.data)
                else:
                    self.grad += grad * np.ones_like(self.data)

        out.grad_fn = _backward
        out.grad_fn_name = "SumBackward"
        return out

    def relu(self):
        # Apply ReLU: max(0, x)
        out_data = np.maximum(self.data, 0)

        # Create a new tensor for the result
        out = Tensor(out_data, requires_grad=self.requires_grad)
        out.parents = {self}

        if self.requires_grad:
            # Define the backward pass for ReLU
            def _backward(grad):
                # The derivative of ReLU is 1 for positive values, 0 for negative
                relu_grad = (self.data > 0).astype(float)  # Create mask for positive values
                if self.grad is None:
                    self.grad = grad * relu_grad
                else:
                    self.grad += grad * relu_grad

            out.grad_fn = _backward
            out.grad_fn_name = "ReLUBackward"
        return out

    def softmax(self):
        # Apply softmax to logits for numerical stability
        exps = np.exp(self.data)
        sum_exps = np.sum(exps)
        result = exps / sum_exps
        
        out = Tensor(result, requires_grad=self.requires_grad)  # Output tensor
        out.parents = {self}  # Store parent tensors

        if self.requires_grad:
            def _backward(grad):

                # Compute softmax of the input
                # softmax = exps / sum_exps  # Compute softmax
                # Gradient of log-softmax
                # grad_input = grad - np.sum(grad, axis=-1, keepdims=True) * softmax  # Backpropagate

                grad_input = grad

                if self.grad is None:
                    self.grad = grad_input  # Initialize grad if it's None
                else:
                    self.grad += grad_input  # Accumulate gradients if grad already exists

                return grad_input  # Return gradient input for the next layer

            out.grad_fn = _backward  # Store the backward function
            out.grad_fn_name = "LogSoftmaxBackward"

        return out


    # def log(self):
    #     # Handle log of zero by adding a small epsilon
    #     out = Tensor(np.log(self.data + 1e-9), requires_grad=self.requires_grad)
    #     out._prev = {self}

    #     def _backward(grad):
    #         if self.requires_grad:
    #             if self.grad is None:
    #                 self.grad = grad / (self.data + 1e-9)
    #             else:
    #                 self.grad += grad / (self.data + 1e-9)

    #     out.grad_fn = _backward
    #     out.grad_fn_name = "LogBackward"
    #     return out

    def __pow__(self, power):
        out = Tensor(self.data ** power, requires_grad=self.requires_grad)
        out.parents = {self}


        def _backward(grad):
            if self.requires_grad:
                if self.grad is None:
                    self.grad = grad * power * (self.data ** (power - 1))
                else:
                    self.grad += grad * power * (self.data ** (power - 1))

        out.grad_fn = _backward
        out.grad_fn_name = "PowBackward"
        return out

    def __matmul__(self, other):
        
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data @ other.data, requires_grad=self.requires_grad or other.requires_grad)
        out.parents = {self, other}

        def _backward(grad):
            if self.requires_grad:
                if self.grad is None:
                    self.grad = grad @ other.data.T
                else:
                    self.grad += grad @ other.data.T
            if other.requires_grad:
                if other.grad is None:
                    other.grad = self.data.T @ grad
                else:
                    other.grad += self.data.T @ grad

        out.grad_fn = _backward
        out.grad_fn_name = "MatMulBackward"
        return out


    def __repr__(self):
        grad_fn_str = f", grad_fn=<{self.grad_fn_name}>" if self.grad_fn else ""
        return f"Tensor({self.data}, requires_grad={self.requires_grad}{grad_fn_str})"

    def backward(self):
        
        # Start the backward pass if this tensor requires gradients
        if not self.requires_grad:
            raise ValueError("This tensor does not require gradients.")
        
        # Initialize the gradient for the tensor if not already set
        if self.grad is None:
            self.grad = np.ones_like(self.data)  # Start with gradient of 1 for scalar output
            # self.grad = Tensor(self.grad)  # Convert to a tensor
        
        # A stack of tensors to backpropagate through
        to_process = [self]
    
        # Process the tensors in reverse order (topological order)
        while to_process:
            tensor = to_process.pop()
            # If this tensor has a backward function, call it
            
            if tensor.grad_fn is not None:
                # print(f"Backpropagating through {tensor.grad_fn_name}")
                # Pass the gradient to the parent tensors
                tensor.grad_fn(tensor.grad)
                # Add the parents of this tensor to the stack for backpropagation
                to_process.extend(tensor.parents)
                
    def detach(self):
        # Create a new tensor that shares the same data but has no gradient tracking
        detached_tensor = Tensor(self.data, requires_grad=False)
        detached_tensor.grad = self.grad  # Retain the gradient (but no computation graph)
        detached_tensor.parents = set()  # Detach from the computation graph
        detached_tensor._grad_fn = None  # Remove the function responsible for backward
        detached_tensor._grad_fn_name = None
        return detached_tensor