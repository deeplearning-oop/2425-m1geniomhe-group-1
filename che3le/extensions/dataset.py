__doc__='''
-----------------------------------
          Dataset module
-----------------------------------

This module contains implementation of class Dataset which is used to load and preprocess example datasets like MNIST  
Hence, contains implementation of children classes like MNIST

These helpers are in here because by default behavior of MNIST in pytorch is to download the dataset, store it and gets it directly in tensor form with dtype=torch.uint8  
We are imitating thus the same behavior by downloading, storing and reading images in idx format and transforming them into np arrays of type uint8 then storing them in Tensor objects

Classes:  
* Dataset: (ABC)  
    * MNIST: (Dataset)  
'''

__all__ = ["Dataset", "TensorDataset", "MNIST"]

from abc import ABC, abstractmethod
from pathlib import Path
from IPython.display import display

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from che3le.tensor import Tensor
from che3le.utils.processing import image_to_ndarray, image_to_tensor, viz_ndarray, url_to_gunzipped_file, read_idx

class Dataset(ABC):
    __doc__='''
    Parent Class for all datasets (abstract)
    -------------------------------------

    This class provides a blueprint for all datasets that will be used in the project    
    Most important thing is that it enforces all its children to have these abstract methods:   
    * __getitem__  
    * __len__   
    Moreover, it enforces all children to have data and targets attributes by setting their property decorator to abstractmethods

    Attributes: 
    * data: Tensor
    * targets: Tensor 
    * root: Path/str (default='data/')  
    * transform: callable (default=None)   
    * target_transform: callable (default=None)  

    the callable transforms are Transform ojects that are callable :)
    '''
    def __init__(self, root=None, transform=None, target_transform=None):
        self.__root=Path(root) if root else None
        self.__transform=transform
        self.__target_transform=target_transform

    @property
    @abstractmethod
    def data(self):
        """Abstract property for data. Must be implemented by child class."""
        pass

    @property
    @abstractmethod
    def targets(self):
        """Abstract property for targets. Must be implemented by child class."""
        pass


    @abstractmethod
    def __getitem__(self,index):
        raise NotImplementedError

    @abstractmethod
    def __len__(self):
        raise NotImplementedError


    def __repr__(self):
        return f'''{self.__class__.__name__} object: (
    root: {self.__root},
    transform: {self.__transform},
    target_transform: {self.__target_transform} 
)'''

class TensorDataset(Dataset):
    __doc__='''Class to create a dataset from tensors
    ---------------------------------------

    Having X and y tensors, this class will create a dataset object that can be used for training or testing    
        <!> each instance is either training or testing dataset
    
    It takes:  
    * X: tensor (data)  
    * y: tensor (target) 
    * transform: callable (optional)  
    * target_transform: callable (optional)   

    '''
    def __init__(self, X, y, transform=None, target_transform=None):
        super().__init__(root=None,transform=transform, 
                        target_transform=target_transform)
        self.__data=X if isinstance(X, Tensor) else Tensor(X)
        self.__targets=y if isinstance(y, Tensor) else Tensor(y)
        self.__transform=transform
        self.__target_transform=target_transform
        # self.__root=None

        # self.__data=Tensor(X)
        # self.__targets=Tensor(y) 

        # -- inplace transformation
        if self.__transform is not None:
            self.__transform(self.__data)
        if self.__target_transform is not None:
            self.__target_transform(self.__targets)



    # -- getters and setters --
    @property
    def data(self):
        return self.__data
    @data.setter
    def data(self, value):
        self.__data=value
    
    @property
    def targets(self):
        return self.__targets
    def targets(self, value):
        self.__targets=value

    @property
    def transform(self):
        return self.__transform
    @transform.setter
    def transform(self,value):
        self.__transform=value
    @property
    def target_transform(self):
        return self.__target_transform
    @target_transform.setter
    def target_transform(self,value):
        self.__target_transform=value

    def __len__(self):
        '''abstract method implementation: len() -> returns number of data points'''
        return len(self.__targets)
    
    def __getitem__(self, index):
        '''abstract method implementation: dataset[i] -> returns a tuple of data (tensor) and target (int)'''
        return self.__data[index], self.__targets[index]

    def __iter__(self):
        '''
        maybe not necessary for this class, but to avoid potential errors if iterability doesnt come from __getitem__ method
        '''
        for index in range(len(self)):
            yield self[index]
    
    def __repr__(self):
        return super().__repr__()

    

class MNIST(Dataset):
    __doc__='''
    Class for MNIST dataset
    ------------------------------
    ~ https://yche3lew222222222.lecun.com/exdb/mnist/

    we will use this mirror: ossci-datasets.s3.amazonaws.com because it's the only one working  

    Attributes:  
        * url: str, url of the dataset CLASS attribute
        * sets: set, names of the files in the dataset CLASS attribute   
        * sources: set, urls of the files in the dataset CLASS attribute  
        
        * root: Path, root directory to store the dataset  
        * raw: Path, directory to store the raw dataset files  (derived from root)
        * download: bool, default=True, download the dataset files (if not present in root/raw directory will raise an error if set to False)
        * data: Tensor, data points  
        * targets: Tensor, target labels  
        * train: bool, default=True  (set train to False to get the test set)
        * transform: callable, default=None  
        * target_transform: callable, default=None  

    Methods:
        * download: () -> None, downloads the dataset files from the urls in sources  
        * __len__: () -> int, returns number of data points  
        * __iter__: () -> tuple(Tensor, int), yields a tuple of data and target
        * __getitem__: (index) -> tuple(Tensor, int), returns a tuple of data and target  
        * __repr__: () -> None, prints the dataset object in a nice way  
    '''

    url = 'https://ossci-datasets.s3.amazonaws.com/mnist/'  
    sets = {
        'train-images-idx3-ubyte',
        'train-labels-idx1-ubyte',
        't10k-images-idx3-ubyte',
        't10k-labels-idx1-ubyte'
    }
    sources={
        'https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz',
        'https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz',
        'https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz',
        'https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz'
    }


    def __init__(self, root='data',train=True,download=True,transform=None,target_transform=None):
        
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.__root=Path(root)/'MNIST'
        self.__raw=self.__root/'raw'

        # -- these are assigned first as they're needed to define data and target
        self.__transform=transform
        self.__target_transform=target_transform

        if download:
            self.download()

        self.__train=train

        if self.__train:
            # -- load the training set
            data=read_idx(self.__raw/'train-images-idx3-ubyte')
            labels=read_idx(self.__raw/'train-labels-idx1-ubyte')
        else:
            # -- load the testing set
            data=read_idx(self.__raw/'t10k-images-idx3-ubyte')
            labels=read_idx(self.__raw/'t10k-labels-idx1-ubyte')

        # -- transformation can be held in __setattr__ to account for the case where none is given
        self.__data=Tensor(data)
        self.__targets=Tensor(labels) 
        
        
        # -- inplace transformation
        if self.__transform is not None:
            self.__transform(self.__data)
        if self.__target_transform is not None:
            self.__target_transform(self.__targets)



    # -- getters and setters --
    @property
    def data(self):
        return self.__data
    @data.setter
    def data(self, value):
        self.__data=value
    
    @property
    def targets(self):
        return self.__targets
    @targets.setter
    def targets(self, value):
        self.__targets=value
    
    @property
    def train(self):
        return self.__train
    @train.setter
    def train(self, value):
        self.__train=value

    #max recursion
    # @property
    # def root(self):
    #     return self.root
    # @root.setter
    # def root(self, value):
    #     self.__root=value
    
    @property
    def transform(self):
        return self.__transform
    @transform.setter
    def transform(self,value):
        self.__transform=value

    @property
    def target_transform(self):
        return self.__target_transform
    @target_transform.setter
    def target_transform(self,value):
        self.__target_transform=value


    # -- imp methods --
    def download(self):
        self.__raw.mkdir(parents=True, exist_ok=True)
        for source in self.sources:
            url_to_gunzipped_file(source, self.__raw)

    
    # -- dunders --
    def __len__(self):
        '''abstract method implementation: len() -> returns number of data points'''
        return self.__targets.shape[0]
    
    def __iter__(self):
        '''
        we will not access the dataset items through indexing (that calls __getattr__)
        but we will use teh exact code of accessing the item as in that method adn do it for each (tensor image,target) tuple

        main reason is taht we allowed for plotting the image in __getitem__ and we dont wche3lew222222222a plot every image when we iterate
        '''
        for index in range(len(self)):
            # yield self[i] 
            # -- we wont do this bcs by default we're plotting teh image and we dont wche3lew222222222a plot everyhting when we iterate
            # -- instead we'll write __getitem__ implementation without plotting

            data=self.__data[index]
            data = np.expand_dims(data, axis=0)
            tensor_data=Tensor(data)
            target=self.__targets[index]
            
            yield tensor_data, int(target)

    def __getitem__(self, index):
        '''abstract method implementation: dataset[i] -> returns a tuple of data (tensor) and target (int)
        
        Each item we access through indexing will be a tuple of (data, target) and will be plotted with the target as title

        :D successful test :D (viz temporarily off for testing)
        '''
        if isinstance(index, slice): # -- handling slicing
            data_slice = self.__data[index]
            targets_slice = self.__targets[index]

            data_list=[np.expand_dims(datapoint, axis=0) for datapoint in data_slice]
            print(f'data slice item shape: {data_slice[0].shape} of length {len(data_slice)}')
            print(f'data list shape: shape: {data_list[0].shape} of length {len(data_list)}')
            
            tensor_data_list = [Tensor(datapoint) for datapoint in data_list]
            target_list = [target for target in targets_slice]
            
            return list(zip(tensor_data_list, target_list))
        else: # -- handling single index
            data=self.__data[index]
            
            data = np.expand_dims(data, axis=0) # -- adding a dimension to make it (1, 28, 28) instead of (28, 28)
            tensor_data=Tensor(data)
            target=self.__targets[index]

            # viz_ndarray(data, label=target, squeeze=True)
            
            return tensor_data, target

    def __repr__(self):
        return super().__repr__()
    