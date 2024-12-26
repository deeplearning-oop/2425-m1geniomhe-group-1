# documentation

## Library

The conventional design of the library:  
- classes start with a capital letter  
- functions start with a small letter and snake case, helper functions start with an underscore  
- the library is divided into modules where each module is a python file  
- each file contain one or more classes and functions related to the main class the module is named after  
- documentation for each function, class and module is provided by a top text block surrounded bby triple quotes  
- dependencies and their versions are found in the requirements.txt file  

## Classes

### dtype

This Enum class is within the module `Tensor` and is used to define the data types of the tensor. It has 2 values: `int64` and `float64`. Each dtype object has a string representation and can be accessed by it through `dtype.__members__['int64']` or `dtype.__members__['float64']`for example.

> [!NOTE]
> For this stage of teh library design, we'll start with 2 possible dtypes: `int64` and `float64`. Why? Because these are the primitive datatype datatypes python have when using `int()` or `float()` functions to cast. So to ease out the casting, a next step would be to see how to cast for other dtypes (int32, float32).

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
These variables will be loaded whenever teh Tensor module is imported (which should be the case in all modules of the library). And then, when accessed through our library, it can be accessed this way (exactly like torch):  

```python
>>> import our_torch_lib as torch
>>> torch.int64
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

Need to validate for these types when creating a tensor (could be in `__setattr__` method of Tensor class OR could be handlede within the tensor() function since we want to make the user create the tensor through it, as it is the case in pytorch)

Concerning **dtype**, we will create for now (version 1.0) an Enum with 2 values: `int64` and `float64`. This is because these are the primitive datatypes casted by python when we call either of the `int()` or `float()` functions. We will make these callable in order to perform the datatype casting and we will represent them as strings.

#### Playground: torch functionality

##### DataTypes

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

###### N-D Tensors
When we give it a list of lists, notice that the ndim=number of opened brackets, and the shape is the number of elements in each bracket.  
Othweise if np array, it's easier to process dimensions as we can use the `shape` attribute of the numpy array.  

```python
>>> t = torch.tensor([[1,2,3],[4,5,6]])
>>> print(f'nb of dim:{t.ndim}; dim:{t.shape}; size:{t.size()}')
nb of dim:2; dim:torch.Size([2, 3]); size:torch.Size([2, 3])
```

###### Dunders

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

- _Addition_:  The tensor can be added to another tensor if the shapes are the same, otherwise it raises a `RuntimeError`. If one of them has float, the resulting tensor will have dtype of float.  

```python
>>> t1 = torch.tensor([1,2,3])
>>> t2 = torch.tensor([1.0,2,3])
>>> t1 + t2
tensor([2., 4., 6.])
```
- _Multiplication_:  The tensor can be multiplied by another tensor if the shapes can perform matrix multiplication ((m,n) * (n,p) = (m,p)), otherwise it raises a (????). If one of them is a scalar, it will multiply each element of the tensor by the scalar.  

```python
#try some examples here
```

#### Implementation

> [!CAUTION]
> note for self: need to test the class, pay attention to ndim and shape setters not very sure about the restriction if its plausible (not taking value from user), if not, allow *args or even value but do not assign it to the attribute and print out a message of its inablity to assign the value. Also try matrix multiplication and check how cast_dtype works.

* Attributes:  
    - `data`: the data of the tensor (list or numeric)   
    - `dtype`: the datatype of the tensor (dtype object), default will be `float64` because easier to convert int to float and if tehre is one float in the tensor teh dtype should be float (avoid errors, unless we create a mthod to check if all are int then we can set an automatic assignment function)      
    - `shape`: the shape/dimensions of the tensor (list)  
    - `ndim`: the number of dimensions of the tensor (int)   
    - `requires_grad`: whether to calculate the gradient of the tensor (bool), default is False like in pytorch  
    - `is_leaf`: whether the tensor is a leaf in the computation graph (bool), default is True like in pytorch  
* Methods:  
    * Dunders:  
        - `__init__`: instanciates the object of type Tensor  
        - `__repr__`: returns the string representation of the tensor  
        - `__str__`: returns the string representation of the tensor  
        - `__iter__`: returns an iterator of the tensor (for the tensor to be iterable, it should be a tensor of tensors, where the lowest level is a scalar which is a 0D tensor)  **NOT IMPLEMENTED YET**  
        - `__getitem__`: returns the element at the index given in the tensor (from top level list). Fruthermore, getting item from an item should be possible as it is a tensor of tensors. **NOT IMPLEMENTED YET**     
        - `__eq__`: returns a tensor of booleans, where the values are True if the corresponding values in the tensors are equal, otherwise False  **NOT IMPLEMENTED YET**  
        - `__add__`: returns the sum of the tensors if they have the same shape  **NOT IMPLEMENTED YET**  
        - `__mul__`: returns the product of the tensors if they can be multiplied  **NOT IMPLEMENTED YET**  
        - `__sub__`: returns the difference of the tensors if they have the same shape  **NOT IMPLEMENTED YET**  
        - `__setattr__`: sets the value of the attribute of the tensor. Very important, used to validate the data input and the dtype, while validating data, we can also set the shape and ndim attributes.   
        - `__len__`: returns the length of the tensor (number of elements in the tensor, and as `__getitem__`, it should be able to return the length of the tensor of tensors). Raises an error when given a 0D tensor (to match with pytorch functionality)  **NOT IMPLEMENTED YET** 
        **NOT IMPLEMENTED YET**
    * Modulators (getters and setters): Normal implementation of getters and setters but worth noting:  
        - _data setter_: calls `__setattr__` so no need to validate explicitly in the setter, and automcatically assigns the shape and ndim attributes   
        - _dtype setter_: calls `__setattr__` so no need to validate explicitly in the setter  
        - _dtype deleter_: doesnt delete the dtype attribute, but sets it to default `float64`
        - _shape setter_: **does not allow to take shape from user**, it is automatically assigned by the data setter  
        - _ndim setter_: **does not allow to take ndim from user**, it is automatically assigned by the data setter (length of shape)  
        - _requires\_grad and is\_leaf setter_ : makes sure the value is a boolean, otherwise raises a ValueError (handled in try except block) specifying the setter function raising the error, leaving its value to teh boolean value taht was prior to teh last setting attempt.  
        - _requires\_grad and is\_leaf deleter_: sets the value to their respective default values (False and True)
    * Helper methods:  
        * validate_dtype(dt:str)->dtype: validates the dtype to be one enlisted within the enum and returns its dtype object (which is callable). This will be used in `__setattr__` method to validate the dtype.  
        * validate_data(data)->list: validates the data to be a list of numbers or list of lists of numbers and returns its DIMENSIONS (as part of the work to validate uniformity of dimensions and to ease out assingment). This will be used in `__setattr__` method to validate the data.   
        * cast_dtype()->list: performs casting of the data to dtype (as the dtype object is callable by design) and returns the new data. This will also be used in `__setattr__` method to assign the data attribute.  
     


## References

_references related to deep learning, ANNs, pytorch, oop and python module writing_

* [pytorch for DL](https://www.learnpytorch.io/)  

_To load large datasets, need to actually download them in a directory and then load them in the notebook by accessing a deafult path name which we have assigned in the implementation. e.g., download MNIST from web in data/MNIST can save images and labels each in a subdirectory, when we load we actually go through the files and convert them to tensors_

* [CO large datasets download](https://oyyarko.medium.com/google-colab-work-with-large-datasets-even-without-downloading-it-ae03a4d0433e)  
