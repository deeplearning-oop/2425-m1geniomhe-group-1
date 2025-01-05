# Computation Graph

In our library, the computation graph is implicitly constructed by storing the parents of the outputs of all operations along with name of the operation that led to them and its corresponding derivation formula. 

## Forward Pass & Graph Construction

### Initial Version

With the execution of forward through `y_hat = model(x)`, multiple linear steps are executed:
```python 
x = self.linear1(x)
x = self.relu(x)
x = self.linear2(x)
x = self.softmax(x)
```
Each step involves one or more basic operations. For instance, `self.linear1(x)` consists of the \__matmul__ `x@weights` and the \__add__ `(x@weights)+bias`. For each operation, the parents, operation name, and corresponding grad function will be stored in the output. Thus, by the end of the forward propagation, we can trace y_hat as the softmax output of its predecessor, which is the output of the addition of the biases of linear2 and (x@weights2), and so on to the beginning. This gives rise to a conceptual computation graph representation storing all primitive steps of the forward propagation, as well as the gradient functions needed for the backward propagation later on. 

To achieve this, all primitive operations were redifined in Tensor. An example is provided below:

```python
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
```
Within each primitive operation, the operands are added to a set called parents. It must be a set since the same operand could occur more than once but it must be counted only once. Similarly, the operation name is stored in grad_fn_name. A _backward function is defined within each primitive operation to compute its gradient. The function is stored in grad_fn to be onyl executed when needed. 

Moreover, in the initial version (as in branches `prototype` and `prototype2`), the activation functions were also defined in the same format within Tensor to simplify testing. The forward method of the loss functions that inherit class Loss were also defined in the same format to contribute to the computation graph construction. For instance, MSE was defined as follows:

```python
class MSE(Loss):
    def forward(self, y, y_hat):
        batch_size = y_hat.data.shape[0]
        error = y_hat.data - y.data
        loss = np.mean(error ** 2)
        out = Tensor(loss, requires_grad=True)
        
        out.parents = {y, y_hat}

        def _backward(grad):
            grad_input = 2 * (y_hat.data - y.data) / batch_size
            if y_hat.grad is None:
                y_hat.grad = grad_input
            else:
                y_hat.grad += grad_input

        out.grad_fn = _backward
        out.grad_fn_name = "MSEBackward"
        
        return out
```

### Refined Version

This initial version was optimized by seperating the activation functions from Tensor and simplifying the code for all operations through the use of a backward decorator. To this end, a seperate function `grad_compute` was defined to compute the gradients of all operations, as in the snippet below:

```python
def grad_compute(self, grad, op_type, other=None):
        def update_grad(grad, target_tensor, factor=1):
            """Helper function to update the gradient of a tensor."""
            if target_tensor.requires_grad:
                target_tensor.grad = grad * factor if target_tensor.grad is None else target_tensor.grad + grad * factor

        if op_type == "add":
            update_grad(grad, self)
            update_grad(grad, other, factor=1)

        elif op_type == "neg":
            update_grad(-grad, self)

        elif op_type == "mul":
            update_grad(grad * other.data, self)
            update_grad(grad * self.data, other)
```

Furthermore, a `backward_decorator` was defined to take care of the execution as well as the operation name, parents, and gradient function storage for all operation types:

```python
def backward_decorator(op_type):
        def decorator(func):
            def wrapper(self, other):
                result = func(self, other)
                other = other if isinstance(other, Tensor) else Tensor(other)
                result.parents = {self, other}
                # Attach the grad function to the result tensor
                result.grad_fn = lambda grad: self.grad_compute(grad, op_type, other)
                result.grad_fn_name = f"{op_type}Backward"
                return result
            return wrapper
        return decorator
```

This allows for the simplification of the definition of all operations to this format:
```python
 @backward_decorator("matmul")
    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return Tensor(self.__data @ other.__data, requires_grad=self.__requires_grad or other.__requires_grad, is_leaf=False)
```
The decorator takes the forward operation name as a parameter, both to store if in `grad_fn_name` and to assign the proper `grad_fn` using the gradient engine `grad_compute`.

A similar organization was applied to both `Activation` and `Loss`, the main difference being defining the backward_decorator as a statis method within these 2 classes to be able to use in all classes inheriting them. An example of these functions within the class Loss is highlighted below:

```python
    def grad_compute(self, loss_type, y, y_hat):
        """
        #Centralized gradient computation for different loss functions.
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
            y_hat.grad = grad_input if y_hat.grad is None else y_hat.grad + grad_input

        return  y_hat.grad       
```

```python
    @staticmethod
    def backward_decorator(loss_type):
        def decorator(func):
            def wrapper(self, y, y_hat):
                out = func(self, y, y_hat)
                out.grad_fn = lambda grad: self.grad_compute(loss_type, y, y_hat)
                out.grad_fn_name = f"{loss_type}Backward"
                out.parents = {y, y_hat}
                return out
            return wrapper
        return decorator
```
Then MSE is simplified to:
```python
class MSE(Loss):
    @Loss.backward_decorator("MSE")
    def forward(self, y, y_hat):
        error = y_hat.data - y.data
        loss = np.mean(error ** 2)
        return Tensor(loss, requires_grad=True, is_leaf=False)
```

## Backward Pass

The main backward function is defined in `Tensor` to be executed through:
```python
loss = loss_fn(y, y_hat) # loss is the Tensor output of the loss function
loss.backward()
```

It starts by setting the gradient of this first tensor `loss` to 1.
```python
    def backward(self):
        # Start the backward pass if this tensor requires gradients
        if not self.__requires_grad:
            raise ValueError("This tensor does not require gradients.")  
        # Initialize the gradient for the tensor if not already set
        if self.grad is None:
            self.grad = np.ones_like(self.__data)  # Start with gradient of 1 for scalar output
```

### Topological Ordering

Next, a stack data structure is used to ensure following the topological ordering in the backward propagation. 
```python     
        to_process = [self]
        # Processing the tensors in reverse order through a stack data structure (to establish topological order)
        while to_process:
            tensor = to_process.pop()
            # If this tensor has a backward function, then call it
            if tensor.grad_fn is not None:
                tensor.grad_fn(tensor.grad) # Pass the gradient to the parent tensors
                # Add the parents of this tensor to the stack for backpropagation
                to_process.extend([parent for parent in tensor.parents if parent.requires_grad])
```
The tensor `loss` is the first to be added to `to_process`. Next, its `grad_fn` is executed and its parents (y and y_hat) are added to the stack. The same process keeps repeating until the stack is empty. The `grad_fn` of `loss` was defined through the static `backward_decorator` during the forward propagation, but only executed when `backward()` is called.  

Following this approach, the chain rule will be used to compute the gradient of the loss w.r.t the weights and biases by traversing the computation graph step-by-step in reverse order. Each node will execute its previously stored `grad_fn` and add its `parents` to the stack.

A condition is added to correct the biases gradient since the biases are broadcasted in the linear layers computations to match the batch size used. In this condition, `is_leaf` is used to only check for leaf tensors that has no parents (i.e. only weights, biases, and input data). `is_leaf` is set as True by default, but it is set as False for the tensor outputs of all operations.  

```python
          #check if the tensor is a leaf and it was broadcasted (in case of batch_size>1)
          if tensor.is_leaf and tensor.data.shape != tensor.grad.shape:
                tensor.grad = np.sum(tensor.grad,axis=1).reshape(-1,1) #adjust the shape to match the data shape
```
