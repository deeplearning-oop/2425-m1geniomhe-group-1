# documentation

<!-- \usepackage[utf8]{inputenc} -->

_this file is temporary in order to report the steps taken later in the report_

## Library

The conventional design of the library:  
- classes start with a capital letter  
- functions start with a small letter and snake case, helper functions start with an underscore (helper not applied yet)    
- the library is divided into modules where each module is a python file  
- each file contain one or more classes and functions related to the main class the module is named after  
- documentation for each function, class and module is provided by a top text block surrounded by triple quotes  
- dependencies and their versions are found in the requirements.txt file  

## Classes

### dtype

This Enum class is within the module `Tensor` and is used to define the data types of the tensor. It has 2 values: `int64` and `float64`. Each dtype object has a string representation and can be accessed by it through `dtype.__members__['int64']` or `dtype.__members__['float64']`for example.

> [!NOTE]
> For this stage of the library design, we'll start with 2 possible dtypes: `int64` and `float64`. Why? Because these are the primitive datatype datatypes python have when using `int()` or `float()` functions to cast. So to ease out the casting, a next step would be to see how to cast for other dtypes (int32, float32).

> [!TIP]
> next enhancement can also provide a function to enlist the possible values to make it more ui friendly, but this is unnessecary as 1. its exactly equivalent to the one liner accessing it from the `__members__` dictionary and 2. the user is not meant to interact with the dtype object directly, but through the `Tensor` class. One can argue that user want to see what dtypes are available but for now, s/he' ll receive the list when an error is thrown if they assign an unavailable dtype through creating a Tensor object.  

* Dunders:  
    * `__repr__`: returns the string representation of the dtype object  
    * `__str__`: _not implemented, it calls_ `__repr__`  
    * `__call__`: makes the object callable, this way can perform `int()` or `float()` casting on the object through `dtype.int64()` or `dtype.float64()`   

Aliasing was done in the Tensor module to make it a more torch-y way of accessing the dtype object:  
```python
int64 = dtype.int64
float64 = dtype.float64
```
These variables will be loaded whenever the Tensor module is imported (which should be the case in all modules of the library). And then, when accessed through our library, it can be accessed this way (exactly like torch):  

```python
>>> import our_torch_wannabe_lib as torch_wannabe
>>> torch_wannabe.int64
```

### Tensor

This class is within a module `Tensor` which take an input of numerics(scalars, 0D tesnors), ndarrays or multidimensional lists and instanciates an object of type `Tensor`.

> [!NOTE]  
> To create a tensor object, a function `tensor()` is used which is a factory function for the `Tensor` class.

So when creating a tensor, need to 1st check the data type. We can give a list (which is recorgnaized as a seq), numpy array or a numeric (not recognized). Otherwise, to imitate pytorch, we can give an object of type `torch.Size` (in which case it would be names `lib.Size`, we have to create it ourselves) which returns an empty tensor of these dimensions. We will start by making the size of type list.

When instanciating an object of type Tensor, we can give the following arguments as **data**:  

- list of numbers or list of lists (the items at the end must be numbers, it could be list of list of list of list of numbers, etc)  
- numpy array (not explicitly required to provide this functionality by the assignment, but might be useful when importing, tranforming and processing data)  
- numeric (only really required to handle int32, int64, float32, float64), these numeric types should be defined in the library to imitate pytorch ui  

> [!NOTE]
> _did not implement a numpy output yet, even though its way easier to transform a list into numpy and use it as input for tensor but it's like cheating, as it already has all functionalities like iterability, multiplication, transposition... In here we're writing these from scratch with a list datatype_

Need to validate for these types when creating a tensor (could be in `__setattr__` method of Tensor class OR could be handlede within the tensor() function since we want to make the user create the tensor through it, as it is the case in pytorch)

Concerning **dtype**, we will create for now (version 1.0) an Enum with 2 values: `int64` and `float64`. This is because these are the primitive datatypes casted by python when we call either of the `int()` or `float()` functions. We will make these callable in order to perform the datatype casting and we will represent them as strings.

#### Playground: torch functionality

##### dtypes

in torch, dtype like `torch.float32` is not a class, but is of type `torch.dtype` which is of type type (so it's a class).

```python
>>> import torch
>>> type(torch.float32)
torch.dtype  
>>> type(torch.dtype)
type
```

Even though it is an instance of `torch.dtype`, it has no `__dict__` attribute, and is not callable, which makes the design of the `torch.dtype` class a bit ambiguous.


###### Scalars
Scalar has ndim=0, shape (Size) empty, so it's a 0D tensor  
A scalar is defined when we give a single number (so numeric type) to the tensor constructor.  

```python 
>>> scalar=torch.tensor(1)
>>> print(f'nb of dim:{scalar.ndim}; dim:{scalar.shape}; size:{scalar.size()}')
nb of dim:0; dim:[]; size:torch.Size([])
```

##### N-D Tensors
When we give it a list of lists, notice that the ndim=number of opened brackets, and the shape is the number of elements in each bracket.  
Othweise if np array, it's easier to process dimensions as we can use the `shape` attribute of the numpy array.  

```python
>>> t = torch.tensor([[1,2,3],[4,5,6]])
>>> print(f'nb of dim:{t.ndim}; dim:{t.shape}; size:{t.size()}')
nb of dim:2; dim:torch.Size([2, 3]); size:torch.Size([2, 3])
```

##### Dunders

- _Iterability_:  The tensor is an iterable of tensors, where the lowest level that isn't iterable is the 0 dimensional tensor (scalar)  

```python
>>> t = torch.tensor([1,2,3])
>>> t[0]
tensor(1)
```

- _Equality_: returns a tensor of booleans, where the values are True if the corresponding values in the tensors are equal, otherwise False.  

```python
>>> t1 = torch.tensor([1,2,3])
>>> t2 = torch.tensor([1,2,3])
>>> t1 == t2
tensor([True, True, True])
```

##### Operations

- _Addition_:  The tensor can be added to another tensor if the shapes are the same, otherwise it raises a `RuntimeError`. If one of them has float, the resulting tensor will have dtype of float.  

```python
>>> t1 = torch.tensor([1,2,3])
>>> t2 = torch.tensor([1.0,2,3])
>>> t1 + t2
tensor([2., 4., 6.])
```
_same for subtraction_

>[!CAUTION]  
> pay attention to dtype changes that can result from operations, write a function to use in all of these that validates the end dtype of the tensor based on one of the tensors' dtype (an easy/lazy slution is to convert all to float64, or convert to float64 iff one of the tensors is float64)

- _Multiplication_:  The tensor can be multiplied by a tensor of teh same shape in an `element-wise` manner. If one of them is a scalar, it will multiply each element of the tensor by the scalar. If we're havong a multiplication between a vector and a scalar, the scalar will be broadcasted to the vector. Same between a matrix and a scalar, a matrix and a vector.

- _Matrix Multiplication_:  The tensor can be multiplied by another tensor if the shapes can perform matrix multiplication ((m,n) * (n,p) = (m,p)). In pytorch this is possible sing 2 different notations

```python
>>> t1 = torch.tensor([[1,2,3],[4,5,6]])
>>> t2 = torch.tensor([[1,2],[3,4],[5,6]])
>>> t1 @ t2
tensor([[22, 28],
        [49, 64]])
>>> t1.matmul(t2)  
tensor([[22, 28],
        [49, 64]])  
>>> torch.matmul(t1, t2) #also m1.matmul(m2) 
tensor([[22, 28],
        [49, 64]])
```

- _transpose_:  The tensor can be transposed.

For 0d tensors, the transpose is deprecated, equivalent an identity function:  

```python
>>> t = torch.tensor(1)
>>> t.T
tensor(1)
/tmp/ipykernel_3772/88739345.py:3: UserWarning: Tensor.T is deprecated on 0-D tensors. This function is the identity in these cases. (Triggered internally at ../aten/src/ATen/native/TensorShape.cpp:3691.)
```

Also deprecated for vectors.

> [!IMPORTANT]
> This T function returns an object of the same shape for both 0D and 1D tensors.  

It works fine for 2+D tensors. In higher dimensions, the transpose is the same as the numpy transpose. 

#### Implementation

> [!CAUTION]
> note for self: need to test the class, pay attention to ndim and shape setters not very sure about the restriction if its plausible (not taking value from user), if not, allow *args or even value but do not assign it to the attribute and print out a message of its inablity to assign the value. Also try matrix multiplication and check how cast_dtype works. Also check out comments in the code, left some questions to answer

* Attributes:  
    - `data`: the data of the tensor (list or numeric)   
    - `dtype`: the datatype of the tensor (dtype object), default will be `float64` because easier to convert int to float and if there is one float in the tensor the dtype should be float (avoid errors, unless we create a mthod to check if all are int then we can set an automatic assignment function)      
    - `shape`: the shape/dimensions of the tensor (list)  
    - `ndim`: the number of dimensions of the tensor (int)   
    - `requires_grad`: whether to calculate the gradient of the tensor (bool), default is False like in pytorch  
    - `is_leaf`: whether the tensor is a leaf in the computation graph (bool), default is True like in pytorch  
* Methods:  
    * **Dunders**:  
        - [x] `__init__`: instanciates the object of type Tensor  
        - [x] `__repr__`: returns the string representation of the tensor  
        - [x]  `__str__`: returns the string representation of the tensor like in pytorch (Tensor(1.0) for example)  
        - [x] `__iter__`: returns an iterator of the tensor (for the tensor to be iterable, it should be a tensor of tensors, where the lowest level is a scalar which is a 0D tensor)  
        - [x] `__getitem__`: returns the element at the index given in the tensor (from top level list). Fruthermore, getting item from an item should be possible as it is a tensor of tensors.     
        - [ ] `__eq__`: returns a tensor of booleans, where the values are True if the corresponding values in the tensors are equal, otherwise False  **NOT IMPLEMENTED YET**  
        - [x] `__add__`: returns the sum of the tensors if they have the same shape  
        - [x] `__sub__`: returns the difference of the tensors if they have the same shape    
        - [x] `__mul__`: returns the product of the tensors if they can be multiplied    
        - [x] `__setattr__`: sets the value of the attribute of the tensor. Very important, used to validate the data input and the dtype, while validating data, we can also set the shape and ndim attributes.   
        - [x] `__len__`: returns the length of the tensor (number of elements in the tensor, and as `__getitem__`, it should be able to return the length of the tensor of tensors). Raises an error (TypeError) when given a 0D tensor (to match with pytorch functionality).      
    * **Modulators** (getters and setters): Normal implementation of getters and setters but worth noting:  
        - _data setter_: calls `__setattr__` so no need to validate explicitly in the setter, and automcatically assigns the shape and ndim attributes   
        - _dtype setter_: calls `__setattr__` so no need to validate explicitly in the setter. **Very important: when setting the dtype, it should cast the data to the new dtype, this is done in setter and not is `__setattr__` as dtype is the first attribute to be set when instanciating the object** (due to control flow and making sure we have dtype when `__setattr__` is called on data to cast it properly). Hence, it would give an error if we perform casting in `__setattr__` for a new instance of Tensor.  
        - _dtype deleter_: doesnt delete the dtype attribute, but sets it to default `float64`
        - _shape setter_: **does not allow to take shape from user**, it is automatically assigned by the data setter  
        - _ndim setter_: **does not allow to take ndim from user**, it is automatically assigned by the data setter (length of shape)  
        - _requires\_grad and is\_leaf setter_ : makes sure the value is a boolean, otherwise raises a ValueError (handled in try except block) specifying the setter function raising the error, leaving its value to the boolean value taht was prior to the last setting attempt.  
        - _requires\_grad and is\_leaf deleter_: sets the value to their respective default values (False and True)
    * **Helper methods** _(static methods)_ :  
        * validate_dtype(dt:str)->dtype: validates the dtype to be one enlisted within the enum and returns its dtype object (which is callable). This will be used in `__setattr__` method to validate the dtype.  
        * validate_data(data:list or num)->list: validates the data to be a list of numbers or list of lists of numbers and returns its DIMENSIONS (as part of the work to validate uniformity of dimensions and to ease out assingment). This will be used in `__setattr__` method to validate the data.   
        * cast_dtype(l:list, dt:dtype)->list: performs casting of the data to dtype (as the dtype object is callable by design) and returns the new data. This will also be used in `__setattr__` method to assign the data attribute.   
    * **Math**:  
        * [x] `T`: returns the transpose of the tensor. Since this mathematical function performs the same transpose for 2+D numpy arrays, we will use numpy transpose. Implementing this function from scratch wont be as computationally feasible/efficient since numpy is known for its speed in matrix operations (it is written in C).

> [!NOTE]  
> ENCAPSULATION PORTRAYED. Future enhancement for helper methods is to make them outside the class (no longer accessible through the class name). The reason why its designed this way though is that this functionality is only used within the class and is not meant to be used outside of it. Static is important because the functionality belonds to the class and not to the instance. Maybe a minor enhacement would still keep it static but make them private (by adding an underscore before the name) to make it clear that they are not meant to be used by users nor by the class itself

### Dataset

This class is within a module `Dataset` which is used to load datasets. It is a parent class for all datasets that will be loaded in the library like MNIST.

First an overview on how it works in pytorch:

The `torchvision` library contains the modules datasets, dataloaders and transforms.  

`MNIST` is designed as a child of datasets.  
When loaded, it will download the MNIST dataset and store it within the working directory (here we're in `tests/notebooks/` of the main repo). The following directories and files will be created t10k-images-idx3-ubyte     t10k-labels-idx1-ubyte     train-images-idx3-ubyte     train-labels-idx1-ubyte
t10k-images-idx3-ubyte.gz  t10k-labels-idx1-ubyte.gz  train-images-idx3-ubyte.gz  train-labels-idx1-ubyte.gz

```text
notebooks/ # -- the wd of this notebook --
└── data/
    └── MNIST/
        └── raw/
            ├── t10k-images-idx3-ubyte  
            ├── t10k-labels-idx1-ubyte
            ├── train-images-idx3-ubyte
            ├── train-labels-idx1-ubyte
            ├── t10k-images-idx3-ubyte.gz
            ├── t10k-labels-idx1-ubyte.gz
            ├── train-images-idx3-ubyte.gz
            └── train-labels-idx1-ubyte.gz
```
     


## References

### tutorials

_references related to deep learning, ANNs, pytorch, oop and python module writing_

#### for nn from scratch:

- [neural networks from scratch: math + python by The Independent Code (youtube)](https://www.youtube.com/watch?v=pauPCy_s0Ok&list=WL&index=75)  
    Some useful formulae (for backpropagation):  
    $$\frac{\partial E}{\partial W} = \frac{\partial E}{\partial Y} X^T; \frac{\partial E}{\partial B} = \frac{\partial E}{\partial Y};\frac{\partial E}{\partial X} = W^T \frac{\partial E}{\partial Y}$$  

    ```python 
    from layer import Layer
    import numpy as np

    class Dense(Layer):
        def __init__(self, input_size, output_size):
            # Initialize weights and biases with random values
            self.weights = np.random.randn(output_size, input_size)
            self.bias = np.random.randn(output_size, 1)
        
        def forward(self, input):
            # Forward pass: Y = W * X + B
            self.input = input
            return np.dot(self.weights, self.input) + self.bias
        
        def backward(self, output_gradient, learning_rate):
            '''input taken: ∂E/∂Y (output_gradient) and learning_rate
            output: ∂E/∂X (input_gradient) to propagate backward

            3 steps:  
            1. Compute weight gradient: ∂E/∂W = ∂E/∂Y ⋅ X^T  
            2. update parameters: W = W - learning_rate * ∂E/∂W; B = B - learning_rate * ∂E/∂B  
            3. Compute input gradient to propagate backward: ∂E/∂X = W^T ⋅ ∂E/∂Y
            '''

            #1. compute weight gradient: ∂E/∂W = ∂E/∂Y ⋅ X^T
            weights_gradient = np.dot(output_gradient, self.input.T)
            
            #2. update param
            # -- Update weights: W = W - learning_rate * ∂E/∂W
            self.weights -= learning_rate * weights_gradient
            # -- Update biases: B = B - learning_rate * ∂E/∂B; bias gradient: ∂E/∂B = ∂E/∂Y (bias gradient equals the output gradient directly)
            self.bias -= learning_rate * output_gradient
            
            #3. compute input gradient to propagate backward: ∂E/∂X = W^T ⋅ ∂E/∂Y
            return np.dot(self.weights.T, output_gradient)
    ```
    > [!IMPORTANT]  
    > In reality, in forward pass, it has an activation function, so it's `f(WX + B)`, but here we're doing only the linear part (`WX + B`)  
    > The activation function will be implemented in the next layer (implemented in next step), output will have the same shape as the input.  
    [_figma board for drawings_](https://www.figma.com/board/KH2skMxUOiDIe27AOMbkrd/Untitled?node-id=0-1&p=f&t=MQ4mh3ljMjb1qY39-0)

    $$\frac{\partial E}{\partial X}=\frac{\partial E}{\partial Y} \odot f'(X)$$
    _this is done in one line through `np.multiply`_ 

    ```python
    from layer import Layer
    import numpy as np

    class Activation(Layer):
        def __init__(self, activation, activation_prime):
            self.activation = activation  # The activation function f(X)
            self.activation_prime = activation_prime  # The derivative of the activation function f'(X)

        def forward(self, input):
            self.input = input
            return self.activation(self.input)  # Apply the activation function f(X)

        def backward(self, output_gradient, learning_rate):
            # ∂E/∂X = ∂E/∂Y ⊙ f'(X)
            return np.multiply(output_gradient, self.activation_prime(self.input))  # Element-wise multiplication
    ```

    _e.g. on implementing known activation functions_  
    ```python
    from activation import Activation
    import numpy as np

    class Tanh(Activation):
        def __init__(self):
            tanh = lambda x: np.tanh(x)  
            tanh_prime = lambda x: 1-np.tanh(x)**2  
            super().__init__(tanh, tanh_prime)  
    ```        

    for the error, taking MSE (mean squared error) as example:  
    $$E = \frac{1}{2} \sum_{i=1}^{n} (y_i - y^*_i)^2; \frac{\partial E}{\partial Y} = \frac{2}{n}(Y - Y^*)$$  

    ```python
    import numpy as np

    def mse(y_true, y_pred):
        return np.mean(np.power(y_true - y_pred, 2))

    def mse_prime(y_true, y_pred):
        return 2 * (y_pred - y_true) / y_true.size
    ```

    Simple examples to test the implementation:  
    * XOR: input(x1, x2) -> hidden(3 neurons - arbitrary but doen in practice) -> output(1 neuron)  (interesting bcs not linearly separable, need a non linear function to solve it)  
    [check its code at this instant](https://youtu.be/pauPCy_s0Ok?si=ysfz0PYwwEzkoEI2&t=1677)


#### for libraries writing (+doc on github):  

- [private methods in python](https://www.datacamp.com/tutorial/python-private-methods-explained)  
- [python packaging](https://packaging.python.org/en/latest/tutorials/packaging-projects/)  
- [python app in github (by github)](https://docs.github.com/en/actions/use-cases-and-examples/building-and-testing/building-and-testing-python)   
- [python package on github tutorial](https://qbee.io/docs/tutorial-github-python.html)   
- [github docum: diagrams in markdown](https://docs.github.com/en/get-started/writing-on-github/working-with-advanced-formatting/creating-diagrams)  

#### for pytorch:

* [pytorch documentation](https://pytorch.org/docs/stable/index.html)  
* [pytorch github repo](https://github.com/pytorch/pytorch)
* [pytorch for DL](https://www.learnpytorch.io/)  

#### for datasets: 

> _To load large datasets, need to actually download them in a directory and then load them in the notebook by accessing a deafult path name which we have assigned in the implementation. e.g., download MNIST from web in data/MNIST can save images and labels each in a subdirectory, when we load we actually go through the files and convert them to tensors_

* [CO large datasets download](https://oyyarko.medium.com/google-colab-work-with-large-datasets-even-without-downloading-it-ae03a4d0433e)   
* [MNIST official database to accesss](https://yann.lecun.com/exdb/mnist/)



#### tools

- [figma](https://www.figma.com/) for designing images in explanation of the library design (figboard for this project is [here](https://www.figma.com/board/KH2skMxUOiDIe27AOMbkrd/Untitled?node-id=0-1&p=f&t=MQ4mh3ljMjb1qY39-0)) 
- [diagrams.net](https://app.diagrams.net/) for designing the class diagrams of the library  
- [carbon](https://carbon.now.sh/) for designing the code snippets in the documentation  