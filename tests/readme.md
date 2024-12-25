# Wiki

## Library

The conventional design of the library:  
- classes start with a capital letter  
- functions start with a small letter and snake case, helper functions start with an underscore  
- the library is divided into modules where each module is a python file  
- each file contain one or more classes and functions related to the main class the module is named after  
- documentation for each function, class and module is provided by a top text block surrounded bby triple quotes  
- dependencies and their versions are found in the requirements.txt file  

## Modules

### Tensor

This module contains a class `Tensor` which take an input of numerics(scalars, 0D tesnors), ndarrays or multidimensional lists and instanciates an object of type `Tensor`.

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

#### DataTypes

in torch, dtype like `torch.float32` is not a class, but is of type `torch.dtype` which is of type type (so it's a class).

```python
>>> import torch
>>> type(torch.float32)
torch.dtype  
>>> type(torch.dtype)
type
```

Even though it is an instance of `torch.dtype`, it has no `__dict__` attribute, and is not callable, which makes the design of the `torch.dtype` class a bit ambiguous.


##### Scalars
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

- _Addition_:  The tensor can be added to another tensor if the shapes are the same, otherwise it raises a `RuntimeError`. If one of them has float, the resulting tensor will have dtype of float.  

```python
>>> t1 = torch.tensor([1,2,3])
>>> t2 = torch.tensor([1.0,2,3])
>>> t1 + t2
tensor([2., 4., 6.])
```



## References

_references related to deep learning, ANNs, pytorch, oop and python module writing_

* [pytorch for DL](https://www.learnpytorch.io/)  

_To load large datasets, need to actually download them in a directory and then load them in the notebook by accessing a deafult path name which we have assigned in the implementation. e.g., download MNIST from web in data/MNIST can save images and labels each in a subdirectory, when we load we actually go through the files and convert them to tensors_

* [CO large datasets download](https://oyyarko.medium.com/google-colab-work-with-large-datasets-even-without-downloading-it-ae03a4d0433e)  
