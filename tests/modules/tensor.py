'''
---------------------------------
tensor module
---------------------------------
This module contains the implementation of the Tensor class, which is the core data structure used in the deep learning framework  
The Tensor class is designed to mimic the behavior of PyTorch's tensor class, allowing for easy manipulation of data and automatic gradient computation through backpropagation (computation graph)

The module also relies on an Enum class implementation of DataType, to allow to have a dtype attribute in Tensor of type Datatype
which consists of:  
    * int32
    * int64  
    * float32  
    * float64  
    * uint8
p.s. these types will be aliased so tehy can be accessed by tehir names from all over the library,
    and they will be used to convert data to the specified data type through numpy

```python
>>> import ouur_library as torch_wannabe
>>> x=torch_wannabe.Tensor([1,2,3],dtype=torch_wannabe.float32)
# design exaclty identical to pytorch's ui
```

More details will be explained in class implementations

'''

import numpy as np
from enum import Enum

class DataType(Enum):
    '''----------------- Data Types -----------------
    ---------------------------------------------------
    this Enum class is used to define the data types of the tensors,
    it allows restricting the data types to the ones defined in the class
    as well as easily converting the data types to numpy data types

    Mechanism of action relies on:  
        * access of dtype calue from a string representation of the data type
        * __call__ method that allows conversion of data to the specified data type through numpy
    
        p.s. uint8 is used for images as pixel values are in the range [0,255] which fits exactly to 8 bits (2^8=256)  
            => uint8 is the first unit of conversion from image to tensor and a memory saver
            
    ____________________________________________________________________________________________________________________________

    '''

    int32='int32'
    int64='int64'
    float32='float32'
    float64='float64'
    uint8='uint8'

    def __repr__(self):
        return self.name
    def __str__(self):
        return self.value
    def __call__(self, x):
        '''
        >>> DataType.float32([1,2,3])
        array([1., 2., 3.], dtype=float32)
        >>> dtype=DataType.int32  
        >>> dtype(1.7)
        array(1, dtype=int32)
        '''
        return np.array(x, dtype=self.value)
    
    def __eq__(self, other):
        return self.value==other

# ------- aliasing to make them easily accessible --------
int32=DataType.int32
int64=DataType.int64
float32=DataType.float32
float64=DataType.float64
uint8=DataType.uint8

def validate_boolean(value, default=False):
    '''takes a value makes sure its boolean, if its not sets it to default'''
    if not isinstance(value, bool):
        print(f'<!> invalid boolean, setting to {default}')
        value=default
    return value

def validate_data_type(value, default=float32):
    '''takes a value makes sure its a DataType, if its not sets it to default'''
    datatypes=list(DataType.__members__.values())
    datatypes_str=[str(i) for i in datatypes]
    # print(datatypes)
    if isinstance(value, str):
        if value not in datatypes_str:
            print(f'<!> invalid data type, setting to {default}')
            value=default
        else:
            value=DataType[value]
    if value not in datatypes:
        print(f'<!> invalid data type, setting to {default}')
        value=default

    return value


def validate_non_int(value):
    '''takes a value makes sure its not an int, return boolean
            RuntimeError: Only Tensors of floating point and complex dtype can require gradients
'''
    if value==int32 or value==int64:
        raise RuntimeError('Only Tensors of floating point and complex dtype can require gradients')
    return True



class Tensor:
    def __init__(self, data, requires_grad=False, is_leaf=False, dtype=float32):    
        ''' Tesnor constructor:
        
        * respecting encapsulation (private attributes + getters and setters)  
        * handling data type conversion through firstly assigning dtype and handling the rest in __setattr__  
        * default values for requires_grad and is_leaf are set
        
        Note: default dtype will be float32 (unlike torch which is an int)  
            and that's to avoid possibel errors if requires_grad is set to True

        e.g. on error that should be given when type is int and requires_grad is True:  

        ```python
        >>> a=torch.tensor([1,2,3], requires_grad=True)
        RuntimeError: Only Tensors of floating point and complex dtype can require gradients
        ```
        '''   
        self.__dtype = validate_data_type(dtype)     
        self.__data = np.array(data) if not isinstance(data, np.ndarray) else data
        self.__requires_grad = requires_grad
        self.__is_leaf = is_leaf
        self.grad = None
        self.grad_fn = None
        self.grad_fn_name = None
        self.parents = set()

    # -- getters and setters
    # -- not initialized but derived:
    @property
    def shape(self):
        return self.__data.shape
    def ndim(self):
        return self.__data.ndim

    @property
    def dtype(self):
        '''
        # direct access from data, 
        # since this getter can only be called after instanciating 
        # we will surely have data assigned
        # ensures dynamic real-time access to data type
        # most importantly: returns a DataType object
        '''
        # print('testing to see if it works')
        return DataType[(self.__data.dtype)]
    @dtype.setter
    def dtype(self, value):
        # data conversion is handlede here bcs when instanciating will go to setattr and data would nto be defined yet, so better do casting here
        self.__dtype = value
        self.__data=value(self.__data)
    @property
    def data(self):
        return self.__data
    @data.setter
    def data(self, value):
        self.__data = value  
    @property
    def requires_grad(self):
        return self.__requires_grad
    @requires_grad.setter
    def requires_grad(self, value):
        self.__requires_grad = value
    @property
    def is_leaf(self):
        return self.__is_leaf
    @is_leaf.setter
    def is_leaf(self, value):
        self.__is_leaf = value

    def __setattr__(self, name, value):
        # if name=='_Tensor__dtype':
        #     # -- add a validate type function here that checks its within DataType and converts it if given a string
            
        #     value=validate_data_type(value)
           

        if name=='_Tensor__requires_grad':
            value=validate_boolean(value, False)
            if value==True:
                # -- can not set it true if data type is int
                validate_non_int(self.__dtype)
                

        super().__setattr__(name, value)
        


    def __len__(self):
        return len(self.__data)


    def __getitem__(self, idx):
        '''important for dataset and dataloaders'''
        return self.__data[idx]

    # ----- to flatten images --------
    def view(self,*args):
        '''same as torch's functionality, it collapses all dimensions into 1
        
        <!> to be tested sperately on example tensors
        '''
        nd_array=self.__data
        reshaped= nd_array.reshape(args)
        t= self
        t.data=reshaped
        return t
    
    def flatten_batch(self):
        '''
        given that a tenosr is a batch of length batch_size, it'll flatten the dimensions while conserving the batch dimension (and transpsoing to match pytorch's behavior)
    
        e.g., it will take (batch_size,1,28,28) and return (784, batch_size) 

        <!> used for testing while batch training images
        '''
        flattened = np.array([img.flatten() for img in self.__data])  # Shape: (32, 784)
        transposed = flattened.T  # Shape: (784, 32)

        self.__data = transposed
        return self
    
    @property
    def shape(self):
        return self.__data.shape

    def __add__(self, other):
        
        other = other if isinstance(other, Tensor) else Tensor(other)
        result = Tensor(self.__data + other.data, requires_grad=self.__requires_grad or other.requires_grad)
        result.parents = {self, other}

        def _backward(grad):
            if self.__requires_grad:
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
        
        result = Tensor(-self.__data, requires_grad=self.__requires_grad)
        result.parents = {self}

        def _backward(grad):
            if self.__requires_grad:
                if self.grad is None:
                    self.grad = -grad
                else:
                    self.grad -= grad

        result.grad_fn = _backward
        result.grad_fn_name = "NegBackward"
        return result

    def __sub__(self, other):
        
        other = other if isinstance(other, Tensor) else Tensor(other)
        result = Tensor(self.__data - other.data, requires_grad=self.__requires_grad or other.requires_grad)
        result.parents = {self, other}
        
        def _backward(grad):
            if self.__requires_grad:
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
            out = Tensor(self.__data * other, requires_grad=self.__requires_grad)
            out.parents = {self}

            def _backward(grad):
                if self.__requires_grad:
                    if self.grad is None:
                        self.grad = grad * other  # Gradient w.r.t. the scalar
                    else:
                        self.grad += grad * other  # Accumulate gradient w.r.t. the scalar

            out.grad_fn = _backward
            out.grad_fn_name = "ScalarMulBackward"
            return out
        
        # Handle the case when 'other' is a Tensor
        if isinstance(other, Tensor):
            out = Tensor(self.__data * other.data, requires_grad=self.__requires_grad or other.requires_grad)
            out.parents = {self, other}

            def _backward(grad):
                if self.__requires_grad:
                    if self.grad is None:
                        self.grad = grad * other.data  # Gradient w.r.t. the other Tensor
                    else:
                        self.grad += grad * other.data  # Accumulate gradient w.r.t. the other Tensor
                if other.requires_grad:
                    if other.grad is None:
                        other.grad = grad * self.__data  # Gradient w.r.t. self Tensor
                    else:
                        other.grad += grad * self.__data  # Accumulate gradient w.r.t. self Tensor

            out.grad_fn = _backward
            out.grad_fn_name = "TensorMulBackward"
            return out


    def __truediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.__data / other.data, requires_grad=self.__requires_grad or other.requires_grad)
        out.parents = {self, other}

        def _backward(grad):
            if self.__requires_grad:
                if self.grad is None:
                    self.grad = grad / other.data
                else:
                    self.grad += grad / other.data
            if other.requires_grad:
                if other.grad is None:
                    other.grad = -grad * self.__data / (other.data ** 2)
                else:
                    other.grad -= grad * self.__data / (other.data ** 2)

        out.grad_fn = _backward
        out.grad_fn_name = "DivBackward"
        return out

    def mean(self):
        out = Tensor(self.__data.mean(), requires_grad=self.__requires_grad)
        out.parents = {self}

        def _backward(grad):
            if self.__requires_grad:
                if self.grad is None:
                    self.grad = grad / self.__data.size
                else:
                    self.grad += grad / self.__data.size

        out.grad_fn = _backward
        out.grad_fn_name = "MeanBackward"
        return out

    def sum(self):
        out = Tensor(self.__data.sum(), requires_grad=self.__requires_grad)
        out.parents = {self}

        def _backward(grad):
            if self.__requires_grad:
                if self.grad is None:
                    self.grad = grad * np.ones_like(self.__data)
                else:
                    self.grad += grad * np.ones_like(self.__data)

        out.grad_fn = _backward
        out.grad_fn_name = "SumBackward"
        return out

    def relu(self):
        # Apply ReLU: max(0, x)
        out_data = np.maximum(self.__data, 0)

        # Create a new tensor for the result
        out = Tensor(out_data, requires_grad=self.__requires_grad)
        out.parents = {self}

        if self.__requires_grad:
            # Define the backward pass for ReLU
            def _backward(grad):
                # The derivative of ReLU is 1 for positive values, 0 for negative
                relu_grad = (self.__data > 0).astype(float)  # Create mask for positive values
                if self.grad is None:
                    self.grad = grad * relu_grad
                else:
                    self.grad += grad * relu_grad

            out.grad_fn = _backward
            out.grad_fn_name = "ReLUBackward"
        return out

    def softmax(self):
        # Apply softmax to logits for numerical stability
        max_logits = np.max(self.__data, axis=0, keepdims=True)  # Shape (1, N)
        exps = np.exp(self.__data - max_logits)
        sum_exps = np.sum(exps, axis=0, keepdims=True)
        result = exps / sum_exps
        # result = np.exp(self.__data) / sum(np.exp(self.__data))
        
        out = Tensor(result, requires_grad=self.__requires_grad)  # Output tensor
        out.parents = {self}  # Store parent tensors

        if self.__requires_grad:
            def _backward(grad):
                
                # Compute softmax of the input
                # softmax = exps / sum_exps  # Compute softmax
                # Gradient of log-softmax
                # grad_input = grad - np.sum(grad, axis=-1, keepdims=True) * softmax  # Backpropagate
                grad_input = result * (grad - np.sum(grad * result, axis=0, keepdims=True))

                if self.grad is None:
                    self.grad = grad_input  # Initialize grad if it's None
                else:
                    self.grad += grad_input  # Accumulate gradients if grad already exists

                return grad  # Return gradient input for the next layer

            out.grad_fn = _backward  # Store the backward function
            out.grad_fn_name = "LogSoftmaxBackward"

        return out


    # def log(self):
    #     # Handle log of zero by adding a small epsilon
    #     out = Tensor(np.log(self.__data + 1e-9), requires_grad=self.__requires_grad)
    #     out._prev = {self}

    #     def _backward(grad):
    #         if self.__requires_grad:
    #             if self.grad is None:
    #                 self.grad = grad / (self.__data + 1e-9)
    #             else:
    #                 self.grad += grad / (self.__data + 1e-9)

    #     out.grad_fn = _backward
    #     out.grad_fn_name = "LogBackward"
    #     return out

    def __pow__(self, power):
        out = Tensor(self.__data ** power, requires_grad=self.__requires_grad)
        out.parents = {self}


        def _backward(grad):
            if self.__requires_grad:
                if self.grad is None:
                    self.grad = grad * power * (self.__data ** (power - 1))
                else:
                    self.grad += grad * power * (self.__data ** (power - 1))

        out.grad_fn = _backward
        out.grad_fn_name = "PowBackward"
        return out

    def __matmul__(self, other):
        
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.__data @ other.data, requires_grad=self.__requires_grad or other.requires_grad)
        out.parents = {self, other}

        def _backward(grad):
            if self.__requires_grad:
                if self.grad is None:
                    self.grad = grad @ other.data.T
                else:
                    self.grad += grad @ other.data.T
            if other.requires_grad:
                if other.grad is None:
                    other.grad = self.__data.T @ grad
                else:
                    other.grad += self.__data.T @ grad

        out.grad_fn = _backward
        out.grad_fn_name = "MatMulBackward"
        return out


    def __repr__(self):
        grad_fn_str = f", grad_fn=<{self.grad_fn_name}>" if self.grad_fn else ""
        return f"Tensor({self.__data}, requires_grad={self.__requires_grad}{grad_fn_str})"

    def backward(self):
        
        # Start the backward pass if this tensor requires gradients
        if not self.__requires_grad:
            raise ValueError("This tensor does not require gradients.")
        
        # Initialize the gradient for the tensor if not already set
        if self.grad is None:
            self.grad = np.ones_like(self.__data)  # Start with gradient of 1 for scalar output
            # self.grad = Tensor(self.grad)  # Convert to a tensor
        
        # A stack of tensors to backpropagate through
        to_process = [self]
        # Process the tensors in reverse order (topological order)
        while to_process:
            tensor = to_process.pop()
            if tensor.is_leaf and tensor.data.shape != tensor.grad.shape:
                tensor.grad = np.sum(tensor.grad,axis=1).reshape(-1,1)

            # If this tensor has a backward function, call it
            if tensor.grad_fn is not None:
                # print(f"Backpropagating through {tensor.grad_fn_name}")
                # Pass the gradient to the parent tensors
                tensor.grad_fn(tensor.grad)
                # print(tensor.grad)
                # Add the parents of this tensor to the stack for backpropagation
                to_process.extend(tensor.parents)
                
    def detach(self):
        # Create a new tensor that shares the same data but has no gradient tracking
        detached_tensor = Tensor(self.__data, requires_grad=False)
        detached_tensor.grad = self.grad  # Retain the gradient (but no computation graph)
        detached_tensor.parents = set()  # Detach from the computation graph
        detached_tensor._grad_fn = None  # Remove the function responsible for backward
        detached_tensor._grad_fn_name = None
        return detached_tensor
    
    def T(self):
        return Tensor(self.__data.T, requires_grad=self.__requires_grad, is_leaf=self.__is_leaf, dtype=self.__dtype)
    
def tensor(data, dtype=float32, requires_grad=False, is_leaf=False):
    '''
    Factory function, generates a tensor instance instead of calling the class,  
    imitates the torch.tensor() function

    ### parameters
    - data: list or numeric  
    - dtype: dtype, default float64  
    - requires_grad: bool, default False  
    - is_leaf: bool, default True  

    ### returns
    - Tensor instance

    ```
    >>> tensor([1,2,3])
    Tensor([1.0, 2.0, 3.0], requires_grad=False)
    ```
    
    '''
    return Tensor(data, dtype=dtype, requires_grad=requires_grad, is_leaf=is_leaf) 

def zeros(shape, dtype=float32, requires_grad=False, is_leaf=True):
    '''
    Factory function, generates a tensor of zeros instead of calling the class,  
    imitates the torch.zeros() function

    ### parameters
    - shape: tuple  
    - dtype: dtype, default float64  
    - requires_grad: bool, default False  
    - is_leaf: bool, default True  

    ### returns
    - Tensor instance

    ```
    >>> zeros((2,3))
    Tensor([[0., 0., 0.],
            [0., 0., 0.]], requires_grad=False)
    ```
    
    '''
    return Tensor(np.zeros(shape), dtype=dtype, requires_grad=requires_grad, is_leaf=is_leaf)

def ones(shape, dtype=float32, requires_grad=False, is_leaf=True):
    '''
    Factory function, generates a tensor of ones instead of calling the class,  
    imitates the torch.ones() function

    ### parameters
    - shape: tuple  
    - dtype: dtype, default float64  
    - requires_grad: bool, default False  
    - is_leaf: bool, default True  

    ### returns
    - Tensor instance

    ```
    >>> ones((2,3))
    Tensor([[1., 1., 1.],
            [1., 1., 1.]], requires_grad=False)
    ```
    
    '''
    return Tensor(np.ones(shape), requires_grad, is_leaf, dtype=dtype)

def randn(shape, dtype=float32, requires_grad=False, is_leaf=True):
    '''
    Factory function, generates a tensor of random numbers instead of calling the class,  
    imitates the torch.randn() function

    ### parameters
    - shape: tuple  
    - dtype: dtype, default float64  
    - requires_grad: bool, default False  
    - is_leaf: bool, default True  

    ### returns
    - Tensor instance

    ```
    >>> randn((2,3))
    Tensor([[0.1, -0.2, 0.3],
            [0.4, -0.5, 0.6]], requires_grad=False)
    ```
    
    '''
    return Tensor(np.random.randn(*shape), dtype=dtype, requires_grad=requires_grad, is_leaf=is_leaf)

def tensor_like(tensor, data, requires_grad=False, is_leaf=False):
    '''
    Factory function, generates a tensor with the same shape as another tensor instead of calling the class,  
    imitates the torch.tensor() function

    ### parameters
    - tensor: Tensor instance  
    - data: list or numeric  
    - requires_grad: bool, default False  
    - is_leaf: bool, default True  

    ### returns
    - Tensor instance

    ```
    >>> a=tensor([1,2,3])
    >>> tensor_like(a, [4,5,6])
    Tensor([4.0, 5.0, 6.0], requires_grad=False)
    ```
    
    '''
    return Tensor(data, requires_grad, is_leaf, dtype=tensor.dtype)
 
def transpose(tensor):
    '''
    generates a transposed tensor instead of calling the class method
    '''
    return tensor.T()