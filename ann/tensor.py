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
        return self.value==other.value

# ------- aliasing to make them easily accessible --------
int32=DataType.int32
int64=DataType.int64
float32=DataType.float32
float64=DataType.float64
uint8=DataType.uint8

def validate_boolean(value, default):
    '''takes a value makes sure its boolean, if its not sets it to default'''
    if not isinstance(value, bool):
        print(f'<!> invalid boolean, setting to {default}')
        value=default
    return value

def validate_data_type(value, default):
    '''takes a value makes sure its a DataType, if its not sets it to default'''
    datatypes=DataType.__members__.values()
    if not isinstance(value, DataType) or (value not in datatypes):
        print(f'<!> invalid data type, setting to {default}')
        value=default
    if isinstance(value, str):
        value=DataType[value]
    return value

def validate_non_int(value):
    '''takes a value makes sure its not an int, return boolean
            RuntimeError: Only Tensors of floating point and complex dtype can require gradients
'''
    if value==int32 or value==int64:
        raise RuntimeError('Only Tensors of floating point and complex dtype can require gradients')
    return True



class Tensor:
    def __init__(self, data, requires_grad=False, is_leaf=True, dtype='float32'):    
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
        self.__dtype = dtype if isinstance(dtype, DataType) else DataType[dtype]     
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
        if name=='_Tensor__dtype':
            # -- add a validate type function here that checks its within DataType and converts it if given a string
            value=validate_data_type(value, float32)
           

        if name=='_Tensor__requires_grad':
            value=validate_boolean(value, False)
            if value==True:
                # -- can not set it true if data type is int
                validate_non_int(self.__dtype)
                

        super().__setattr__(name, value)
        
    def __repr__(self):
        grad_fn_str = f", grad_fn=<{self.grad_fn_name}>" if self.grad_fn else ""
        return f"Tensor({self.__data}, requires_grad={self.__requires_grad}{grad_fn_str})"


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

    def backward(self):
        
        # Start the backward pass if this tensor requires gradients
        if not self.__requires_grad:
            raise ValueError("This tensor does not require gradients.")
        
        # Initialize the gradient for the tensor if not already set
        if self.grad is None:
            self.grad = np.ones_like(self.__data)  # Start with gradient of 1 for scalar output
        
        # A stack of tensors to backpropagate through (to establish topological order)
        to_process = [self]
        # Processing the tensors in reverse order (topologically ordered)
        while to_process:
            tensor = to_process.pop()
            #check if the tensor is a leaf and it was broadcasted (in case of batch_size>1)
            
            if tensor.is_leaf and tensor.data.shape != tensor.grad.shape:
                tensor.grad = np.sum(tensor.grad,axis=1).reshape(-1,1) #adjust the shape to match the data shape

            # If this tensor has a backward function, then call it
            if tensor.grad_fn is not None:
                # print(f"Backpropagating through {tensor.grad_fn_name} function")
                # Pass the gradient to the parent tensors
                tensor.grad_fn(tensor.grad)
                # Add the parents of this tensor to the stack for backpropagation
                to_process.extend([parent for parent in tensor.parents if parent.requires_grad])
                
    

    def grad_compute(self, grad, op_type, other=None):
        # Helper function to update the gradient
        def update_grad(grad, target_tensor, factor=1):
            """Helper function to update the gradient of a tensor."""
            if target_tensor.requires_grad:
                target_tensor.grad = grad * factor if target_tensor.grad is None else target_tensor.grad + grad * factor

        if op_type == "add":
            update_grad(grad, self)
            update_grad(grad, other, factor=1)

        elif op_type == "neg":
            update_grad(-grad, self)

        elif op_type == "sub":
            update_grad(grad, self)
            update_grad(-grad, other)

        elif op_type == "mul":
            update_grad(grad * other.data, self)
            update_grad(grad * self.data, other)

        elif op_type == "matmul":
            update_grad(grad @ other.data.T, self)
            update_grad(self.data.T @ grad, other)

        elif op_type == "div":
            update_grad(grad / other.data, self)
            update_grad(-grad * self.data / (other.data ** 2), other)

        elif op_type == "mean":
            update_grad(grad / self.data.size, self)

        elif op_type == "sum":
            update_grad(grad * np.ones_like(self.data), self)

        elif op_type == "pow":
            update_grad(grad * other * (self.data ** (other - 1)), self)
      
      
    def backward_decorator(op_type):
        def decorator(func):
            def wrapper(self, other):
                result = func(self, other)
                # Set parents of the result tensor
                other = other if isinstance(other, Tensor) else Tensor(other)
                result.parents = {self, other}
                # Attach the grad function to the result tensor
                result.grad_fn = lambda grad: self.grad_compute(grad, op_type, other)
                result.grad_fn_name = f"{op_type}Backward"
                return result
            return wrapper
        return decorator

    @backward_decorator("add")
    def __add__(self, other):
        # Ensure other is always a Tensor
        other = other if isinstance(other, Tensor) else Tensor(other)
        return Tensor(self.__data + other.__data, requires_grad=self.__requires_grad or other.__requires_grad, is_leaf=False)

    @backward_decorator("neg")
    def __neg__(self):
        return Tensor(-self.__data, requires_grad=self.__requires_grad, is_leaf=False)

    @backward_decorator("sub")
    def __sub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return Tensor(self.__data - other.__data, requires_grad=self.__requires_grad or other.__requires_grad, is_leaf=False)

    @backward_decorator("mul")
    def __mul__(self, other):  
        other = other if isinstance(other, Tensor) else Tensor(other)
        return Tensor(self.__data * other.__data, requires_grad=self.__requires_grad or other.__requires_grad, is_leaf=False)

    @backward_decorator("div")
    def __truediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return Tensor(self.__data / other.__data, requires_grad=self.__requires_grad or other.__requires_grad, is_leaf=False)
    
    @backward_decorator("mean")
    def mean(self):
        return Tensor(self.__data.mean(), requires_grad=self.__requires_grad, is_leaf=False)
    
    @backward_decorator("sum")
    def sum(self):
        return Tensor(self.__data.sum(), requires_grad=self.__requires_grad, is_leaf=False)

    @backward_decorator("pow")
    def __pow__(self, power):
        return Tensor(self.__data ** power, requires_grad=self.__requires_grad, is_leaf=False)

    @backward_decorator("matmul")
    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return Tensor(self.__data @ other.__data, requires_grad=self.__requires_grad or other.__requires_grad, is_leaf=False)
              
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