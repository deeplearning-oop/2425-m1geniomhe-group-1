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


## References

_references related to deep learning, ANNs, pytorch, oop and python module writing_

* [pytorch for DL](https://www.learnpytorch.io/)  

_To load large datasets, need to actually download them in a directory and then load them in the notebook by accessing a deafult path name which we have assigned in the implementation. e.g., download MNIST from web in data/MNIST can save images and labels each in a subdirectory, when we load we actually go through the files and convert them to tensors_

* [CO large datasets download](https://oyyarko.medium.com/google-colab-work-with-large-datasets-even-without-downloading-it-ae03a4d0433e)  
