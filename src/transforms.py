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
    - `Standardize`: Standardize the input data, given a tensor -> returns a tensor
    - `MinMaxNormalize`: Normalize the input data to the range [min, max], given a tensor -> returns a tensor
    - `Compose`: Chain several transformations together, given a list of transformations -> returns a transformed dataset


This module contains the following CALLABLE classes:  
    * Compose  
    * Normalize  
    * ToTensor  
    * Standardize
    * MinMaxNormalize


'''

import numpy as np
import cv2
import PIL
from tensor import Tensor

valid_transforms = ['ToTensor', 'Normalize', 'Standardize', 'MinMaxNormalize', 'Compose']


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
    return Tensor(pixels)

def normalize_tensor(tensor, mean=None, std=None):
    """
    Normalizes a tensor using the provided mean and standard deviation, 
    the default behavior if no mean or std is provided is to use the mean and std of the tensor
    (either both present or both not present)
    
    Parameters:
        tensor (np.ndarray or Tensor): The array to be normalized
        mean (float): The mean to normalize with (optional)
        std (float): The standard deviation to normalize with (optional)
    
    Returns:
        np.ndarray: The normalized array
    """
    if mean is None:
        mean = np.mean(tensor.data)
    if std is None:
        std = np.std(tensor.data)
    
    if std == 0:
        raise ValueError("Standard deviation cannot be zero while normalizing")

    if type(tensor) == Tensor:
        tensor = tensor.data
    
    normalized_array = (tensor - mean) / std
    normalized_tensor=Tensor(normalized_array)
    return normalized_tensor

def standardize_tensor(tensor):
    """
    Standardizes a tensor by subtracting the mean and dividing by the standard deviation,
    this is the normalize_tensor function with mean and std set to None (default)
        => output will have a mean of 0 and a standard deviation of 1
    
    Parameters:
        array (np.ndarray): The array to be standardized.
    
    Returns:
        np.ndarray: The standardized array
    """
    return normalize_tensor(tensor)


def min_max_normalize_tensor(tensor, min, max):
    '''
    Normalizes a tensor to the range [min, max]

    Parameters:
        tensor (np.ndarray or Tensor): The array to be normalized.
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

    if type(tensor) == Tensor:
        tensor = tensor.data
    
    normalized_array = (tensor - min) / (max - min)
    normalized_tensor=Tensor(normalized_array)
    return normalized_tensor

class Transform(ABC):
    '''
    ------------------------------
    Transform parent class:  
    -------------------------------

    this abstract class ensures all children implement the __call__ methods
    '''
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, input):
        pass

class Compose(Transform):
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

    @property
    def transforms(self):
        return self.__transforms
    @transforms.setter
    def transforms(self, value):
        self.__transforms = value

    @staticmethod
    def _validate_transforms(transforms):
        '''
        This method validates the transforms attribute to be a list of callable objects  
        preferably the callale objects have to be classes of this module
        '''
        if not isinstance(transforms, list):
            raise TypeError('not a list, transforms must be a list')
        for transform in transforms:
            if type(transform).__name__ not in valid_transforms:
                raise TypeError(f'not a valid transformation, transforms must be one of: {valid_transforms}')

            
        if len(transforms) == 0:
            raise ValueError('no objects, transforms must be a list of callable objects')
            

    def __setattr__(self, name, value):
        if name == '_Compose__transforms':
            try:
                self._validate_transforms(value)
            except Exception as e:
                print(e)
                value=[]
                # -- if valueError it'll be an empty list and no tranformation will be done
                if isinstance(e, ValueError):
                    print(e,'\n     <!> will be an empty list, no transformation will be done')
            super().__setattr__(name, value)

    def __call__(self, dataset):
        '''
        This method is called when the object is called, it applies the transformations to the dataset

        The idea behind this method is to apply a composite transformation to the dataset, following teh mathematical notation of a composite function f o g o h(x) = f(g(h(x)));
        <!> the order of the transformations is important, the first transformation is applied first, then the second and so on

        ```
        >>> transformation=Compose([ToTensor(), Normalize(mean=0.5, std=0.5)])
        >>> train_data=MNIST(root='data', train=True, download=True, transform=transformation)
        ``` 
        
        '''
        for transform in self.transforms:
            print(f'>>> applying {transform}...')
            dataset = transform(dataset) #no need to assign it, it happens inplace, but keeping it liek that for clarity
        print(f'>>> {self.__transforms} applied successfully <<<')
        return dataset
    
    def __repr__(self):
        return f'Compose(transforms={self.__transforms})'  
    
    def __str__(self):
        return f'Transformations: {self.__transforms}'

    
class ToTensor(Transform):
    '''
    ------------------------------
    ToTensor transformer class:
    -------------------------------
    Transforms images or numpy arrays to tensors, the main idea of this is to:  
        * convert an image to a tensor  
        * scale it to the range [0, 1] (/255 for images since the pixel value is of dtype uint8 and is in the range [0, 255])

    Unlike pytorch that uses PIL to load images, we use opencv since it's based on C++ and is faster (comparisons will be done for this purpose)
    
    This class is callable, exampel of usage:
    ```
    >>> transformation=ToTensor()
    >>> train_data=MNIST(root='data', train=True, download=True)
    >>> x=train_data[0][0]
    >>> np.min(x.data), np.max(x.data)
    0, 255
    >>> transformation(x) #inplace transformation
    >>> np.min(x.data), np.max(x.data)
    0.0, 1.0
    ``` 

    In practice, in this library:
    ```
    >>> train_data=MNIST(root='data', train=True, download=True, transform=ToTensor())
    >>> x=train_data[0][0]
    >>> np.min(x.data), np.max(x.data)
    0.0, 1.0
    ```

    successfully tested :D
    '''
    def __init__(self):  
        pass

    def __call__(self, input):  
        '''
        This method is called when the object is called, it converts an image to a tensor and scales it to the range [0, 1]
        '''
        if isinstance(input, Tensor):
            print(' :O already a tensor')
            tensored=input
        elif isinstance(input, np.ndarray):
            tensored= Tensor(input)
        elif isinstance(input, PIL.Image.Image):
            tensored= Tensor(np.array(input))
        else:
            raise TypeError('input must be a tensor, a numpy array or a PIL image')

        # -- scaling the tensor to the range [0, 1]
        tensored.data = tensored.data / 255 #or min_max_normalize_tensor(tensored, 0, 255)
        return tensored
        
    
    def __repr__(self):
        return 'ToTensor()'
    
    
class Normalize(Transform):
    '''
    ------------------------------
    Normalize transform class:
    -------------------------------

    This class is callable too, it takes a tensor and normalizes it

    Attributes:  
        * mean: mean value of the tensor
        * std: standard deviation value of the tensor
        * inplace: boolean, if True, the normalization is done in place

    <!> 
    '''

    def __init__(self, mean=None, std=None, inplace=True):
        self.__mean = mean
        self.__std = std
        self.__inplace = inplace #tested with inplace=True, it works


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
    @property
    def inplace(self):
        return self.__inplace
    @inplace.setter
    def inplace(self, value):
        self.__inplace = value

    #validating dimensions (mean or srd should be a scalar or a tensor with the same dimensions as the input tensor)


    def __call__(self, tensor_input):
        '''
        This method is called when the object is called, it normalizes a tensor
        '''
        normalized_tensor = normalize_tensor(tensor_input, self.__mean, self.__std)
        if self.__inplace:
            tensor_input.data = normalized_tensor.data
            return tensor_input
        return normalized_tensor
        
    
    def __repr__(self):
        return f'Normalize(mean={self.__mean}, std={self.__std}), inplace={self.__inplace})'

class Standardize(Normalize):
    '''
    ------------------------------
    Standardize transform class:
    -------------------------------

    Same as Normalize, but with mean and std set to None by default
    '''

    def __init__(self, inplace=True):
        super().__init__(mean=None, std=None, inplace=inplace)

    def __repr__(self):
        return f'Standardize(inplace={self.inplace})'
    
    def __call__(self, tensor_input):
        super().__call__(tensor_input)

    
class MinMaxNormalize:
    '''
    ------------------------------
    MinMaxNormalize transform class:
    -------------------------------

    This class is callable

    Attributes:  
        * min: minimum value of the tensor
        * max: maximum value of the tensor
        * inplace: boolean, if True, the normalization is done in place

    if no min or max is provided, the min and max of the tensor are used by default
    '''
    def __init__(self, min=None, max=None, inplace=True):
        self.__min = min
        self.__max = max
        self.__inplace = inplace

    @property
    def min(self):
        return self.__min
    @property
    def max(self):
        return self.__max
    @min.setter
    def min(self, value):
        self.__min = value
    @max.setter
    def max(self, value):
        self.__max = value
    @property
    def inplace(self):
        return self.__inplace
    @inplace.setter
    def inplace(self, value):
        self.__inplace = value
    
    def __call__(self, tensor_input):
        '''
        This method is called when the object is called, it normalizes a tensor
        '''
        normalized_tensor = min_max_normalize_tensor(tensor_input, self.__min, self.__max)
        if self.__inplace:
            tensor_input.data = normalized_tensor.data
            return tensor_input
        return normalized_tensor
    
    def __repr__(self):
        return f'MinMaxNormalize(min={self.__min}, max={self.__max}), inplace={self.__inplace})'
    

