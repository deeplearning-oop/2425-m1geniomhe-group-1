'''
---------------------
Tranforms module:
---------------------

Tranformers are applied as a data preprocessing step before feeding the data to the model.  
In Dataset initialization, there are 2 types of transformations:  
        - `transform` which is applied to the input data (X)  
        - `target_transform` which is applied to the target data (y)

    In both cases, we need to provide of the Tranformer callable classes to be applied to the dataset.  
        <!> if several transformations are needed, `Compose` becomes handy to chain them together.

Possibly most useful transformations to consider:
    - `ToTensor`: Converts the input data to a tensor, this can be a combination of different transformations (image-> tensor or numpy array -> tensor)
    - `Normalize`: Normalize the input data, given a tensor -> returns a tensor  


This module contains the following CALLABLE classes:  
    * Compose  
    * Normalize  
    * ToTensor  


'''

import numpy as np
import cv2
import PIL
import tensor


def image_to_ndarray(image_path,grey=False):
    '''
    takes an image path and transforms it inot a numpy array 
    
        if grey -> image is in 2 dimensions
        if not grey -> image is in 3 (rgb channels)

    depends on nympy and cv2 

    :D successful test :D
    '''

    image = cv2.imread(image_path)
    code=cv2.COLOR_BGR2GRAY if grey else cv2.COLOR_BGR2RGB
    pixels=cv2.cvtColor(image, code)
    return pixels

def image_to_tensor(image_path, grey=False):
    '''
    takes an image path and transforms it inot a tensor 
        if grey -> image is in 2 dimensions
        if not grey -> image is in 3 (rgb channels)

    depends on nympy and cv2 
    :D successful test :D
    '''
    pixels=image_to_ndarray(image_path, grey=grey)
    return tensor.Tensor(pixels)

def normalize_tensor(tensor, mean=None, std=None):
    """
    Normalizes a tensor using the provided mean and standard deviation, 
    the default behavior if no mean or std is provided is to use the mean and std of the tensor
    (either both present or both not present)
    
    Parameters:
        array (np.ndarray): The array to be normalized
        mean (float): The mean to normalize with (optional)
        std (float): The standard deviation to normalize with (optional)
    
    Returns:
        np.ndarray: The normalized array
    """
    if mean is None:
        mean = np.mean(tensor)
    if std is None:
        std = np.std(tensor)
    
    if std == 0:
        raise ValueError("Standard deviation cannot be zero while normalizing")

    if type(tensor) == tensor.Tensor:
        tensor = tensor.data
    
    normalized_array = (tensor - mean) / std
    return normalized_array

def standardize_tensor(tensor):
    """
    Standardizes a tensor by subtracting the mean and dividing by the standard deviation,
    this is the normalize_tensor function with mean=0 and std=1
    
    Parameters:
        array (np.ndarray): The array to be standardized.
    
    Returns:
        np.ndarray: The standardized array
    """
    mean=0
    std=1
    return normalize_tensor(tensor, mean, std)


def min_max_normalize_tensor(tensor, min, max):
    '''
    Normalizes a tensor to the range [min, max]

    Parameters:
        tensor (np.ndarray or tensor.Tensor): The array to be normalized.
        min (float): The minimum value to normalize to.
        max (float): The maximum value to normalize to.
    
    Returns:
        np.ndarray: The normalized array
    '''
    if min >= max:
        raise ValueError("min must be less than max")
    
    if min is None:
        min = np.min(tensor)
    if max is None:
        max = np.max(tensor)

    if type(tensor) == tensor.Tensor:
        tensor = tensor.data
    
    normalized_array = (tensor - min) / (max - min)
    return normalized_array

class Compose:
    '''
    ------------------------------
    Compose transformer class:  
    -------------------------------

    This class is callable, it allows to create a pipeline from an ordered series of tranformations to be applied to a dataset through the `__call__` method when given as an attribute

    Parameters:  
        * transforms: list of transformations to be applied to the dataset
            each transformation is a callable object that takes a dataset as input and returns a transformed dataset
    Returns:  
        * dataset: transformed dataset (tensor)
    '''

    def __init__(self, transforms):
        self.__transforms = transforms

    @staticmethod
    def _validate_transforms(transforms):
        '''
        This method validates the transforms attribute to be a list of callable objects  
        preferably teh callale objects have to be classes of this module
        '''
        if not isinstance(transforms, list):
            raise TypeError('not a list, transforms must be a list')
        for transform in transforms:
            if not callable(transform):
                raise TypeError('not callable, transforms must be a list of callable objects')
            
        if len(transforms) == 0:
            raise ValueError('no objects, transforms must be a list of callable objects')
            

    def __setattr__(self, name, value):
        if name == 'transforms':
            try:
                self._validate_transforms(value)
            except Exception as e:
                print(e)
                value=[]
                # -- if valueError it'll be an empty list and no tranformation will be done
                if isinstance(e, ValueError):
                    print('     <!> will be an empty list, no transformation will be done')
            super().__setattr__(name, value)

    def __call__(self, dataset):
        '''
        This method is called when the object is called, it applies the transformations to the dataset

        The idea behind this method is to apply a composite transformation to the dataset, following teh mathematical notation of a composite function f o g o h(x) = f(g(h(x)));
        <!> the order of the transformations is important, the first transformation is applied first, then the second and so on

        ```
        ``` 
        
        '''
        for transform in self.__transforms:
            dataset = transform(dataset)
        return dataset
    
    def __repr__(self):
        return f'Compose(transforms={self.__transforms})'  
    
    def __str__(self):
        return f'Transformations: {self.__transforms}'

    
class ToTensor:
    '''
    ------------------------------
    ToTensor transformer class:
    -------------------------------
    Transforms images or numpy arrays to tensors, the main idea of this is to:  
        * convert an image to a tensor  
        * scale it to the range [0, 1] (/255 for images since the pixel value is of dtype uint8 and is in the range [0, 255])

    Unlike pytorch that uses PIL to load images, we use opencv since it's based on C++ and is faster (comparisons will be done for this purpose)
    
    This class is callable
    '''
    def __init__(self):  
        pass

    def __call__(self, input):  
        '''
        This method is called when the object is called, it converts an image to a tensor and scales it to the range [0, 1]
        '''
        if isinstance(input, tensor.Tensor):
            print(' :O already a tensor')
            tensored=input
        elif isinstance(input, np.ndarray):
            tensored= tensor.Tensor(input)
        elif isinstance(input, PIL.Image.Image):
            tensored= tensor.Tensor(np.array(input))
        else:
            raise TypeError('input must be a tensor, a numpy array or a PIL image')

        # -- scaling the tensor to the range [0, 1]
        tensored.data = tensored.data / 255 #or min_max_normalize_tensor(tensored, 0, 255)
        return tensored
        
    
    def __repr__(self):
        return 'ToTensor()'
    
    
class Normalize:
    '''
    ------------------------------
    Normalize transform class:
    -------------------------------

    This class is callable too, it takes a tensor and normalizes it

    Attributes:  
        * mean: mean value of the tensor
        * std: standard deviation value of the tensor

    <!> still want to validate the dimensions of the mean and std tensors to be the same as the input tensor
    '''

    def __init__(self, mean, std):
        self.__mean = mean
        self.__std = std

    @property
    def mean(self):
        return self.__mean
    @property
    def std(self):
        return self.__std
    @mean.setter
    def mean(self, value):
        self.__mean = value
    @std.setter
    def std(self, value):
        self.__std = value

    #validating dimensions (mean or srd should be a scalar or a tensor with the same dimensions as the input tensor)


    def __call__(self, tensor):
        '''
        This method is called when the object is called, it normalizes a tensor
        '''
        return (tensor - self.__mean) / self.__std
    
    def __repr__(self):
        return f'Normalize(mean={self.__mean}, std={self.__std})'
