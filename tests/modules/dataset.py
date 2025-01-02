'''
Dataset module
---------------

This module contains implementation of class Dataset which is used to load and preprocess example datasets like MNIST

'''

import matplotlib.pyplot as plt
import numpy as np
import cv2
from pathlib import Path
from typing import Union
import os
import gzip
import requests
from typing import Union
import pandas as pd
from IPython.display import display

import tensor 


def image_to_ndarray(image_path,grey=False):
    '''
    takes an image path and transforms it inot a numpy array \
        if grey -> image is in 2 dimensions
        if not grey -> image is in 3 (rgb channels)

    depends on nympy and cv2 \
    :D successful test :D
    '''

    image = cv2.imread(image_path)
    code=cv2.COLOR_BGR2GRAY if grey else cv2.COLOR_BGR2RGB
    pixels=cv2.cvtColor(image, code)
    return pixels

def image_to_tensor(image_path, grey=False):
    '''
    takes an image path and transforms it inot a tensor \
        if grey -> image is in 2 dimensions
        if not grey -> image is in 3 (rgb channels)

    depends on nympy and cv2 \
    :D successful test :D
    '''
    pixels=image_to_ndarray(image_path, grey=grey)
    return tensor.Tensor(pixels)

def viz_ndarray(ndarray, label=None, squeeze=False):
    '''
    takes a multidimensional array of an image and plots it, if label provided makes it a title

    params:  \
    * ndarray: np.ndarray (or tensor)
    * label: str (optional)  
    * squeeze: bool (optional), if True it squeezes a 2D image thats (1, 28, 28) to (28, 28) for instance

    returns: None
    '''
    if type(ndarray)==tensor.Tensor:
        ndarray=ndarray.data #getting data as tensor

    if squeeze:
        ndarray=np.squeeze(ndarray)

    plt.imshow(ndarray, cmap='gray')
    plt.xticks([])
    plt.yticks([])
    if label:
        plt.title(label)
    plt.show()

def url_to_gunzipped_file(url, path):
    '''
    takes url of .gz file,downloads it and extracts it in path directory

    '''
    filename=url.split('/')[-1]
    filepath=path/filename
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
    }

    if filepath.exists():
        print(f'{filepath} already exists')
    else:
        try:
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                with open(filepath, "wb") as f:
                    f.write(response.content)
                print(f"File downloaded successfully as '{filepath}'.")
            else:
                print(f"Failed to download file. Status code: {response.status_code}")
        except Exception as e:
            print(f"An error occurred: {e}")

    filename_no_gz=filename.replace('.gz','')
    filepath_no_gz=path/filename_no_gz


    if filepath_no_gz.exists():
        print(f'{filepath_no_gz} already exists')
    else:
        with open(filepath, 'rb') as f:
            file_content = f.read()
            gunzip_content = gzip.decompress(file_content)
            with open(filepath_no_gz, 'wb') as f:
                f.write(gunzip_content)

def read_idx(file_path):
    """
    reads an IDX file and returns the data as a numpy array
    
    param:\ 
        file_path (str): Path to the IDX file
    
    returns:\ 
        np.ndarray
    """
    with open(file_path, 'rb') as f:
        magic = int.from_bytes(f.read(4), byteorder='big')
        data_type = (magic >> 8) & 0xFF
        num_dims = magic & 0xFF

        dims = [int.from_bytes(f.read(4), byteorder='big') for _ in range(num_dims)]

        data = np.frombuffer(f.read(), dtype=np.uint8)

        data = data.reshape(dims)
        
    return data

#####################
#####################
# testing:
path=Path('data/MNIST')
path.mkdir(exist_ok=True)
#####################
#####################

class Dataset:
    '''
    Parent Class for all datasets
    ------------------------------
    '''
    def __init__(self, root:Union(str, Path)='data/', transform=None, target_transform=None):
        self.__root=root
        self.__transform=transform
        self.__target_transform=target_transform
        
    # def __len__(self):
    #     raise NotImplementedError

    # def __getitem__(self.__index):
    #     raise NotImplementedError

    def __repr__(self):
        return f'{self.__class__.__name__}(root={self.__root}, transform={self.__transform}, target_transform={self.__target_transform})'


class MNIST(Dataset):
    '''
    Class for MNIST dataset
    ------------------------------
    ~ https://yann.lecun.com/exdb/mnist/

    we will use this mirror: ossci-datasets.s3.amazonaws.com
    '''

    url='https://ossci-datasets.s3.amazonaws.com/mnist/'
    sets={
        'train-images-idx3-ubyte',
        'train-labels-idx1-ubyte',
        't10k-images-idx3-ubyte',
        't10k-labels-idx1-ubyte'
    }

    sources={url+s for s in sets}

    def __init__(self, root:Union(str, Path),train=True,download=True,transform=None,target_transform=None):
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.__root=Path(root)/'MNIST'
        self.__raw=self.__root/'raw'
        if download:
            self.__download()

        self.__train=train
        if self.__train:
            data=read_idx(self.__raw/'train-images-idx3-ubyte')
            labels=read_idx(self.__raw/'train-labels-idx1-ubyte')
        else:
            data=read_idx(self.__raw/'t10k-images-idx3-ubyte')
            labels=read_idx(self.__raw/'t10k-labels-idx1-ubyte')

        self.__data=tensor.Tensor(data) #need to make dtype as uint8
        self.__targets=tensor.Tensor(labels) #same

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

    @property
    def root(self):
        return self.root
    @root.setter
    def root(self, value):
        self.__root=value
    
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
        self.__data.shape

    def __repr__(self):
        temp= {
            'class_name': self.__class__.__name__,
            'number_of_datapoints': len(self),
            'directory': self.__root,
            'train': self.__train,
            'split': 'train' if self.__train else 'test',
            'transforms': self.__transform if self.__transform else 'No transforms'
        }
        df=pd.DataFrame(temp, index=[0])
        display(df)
        return ''

    def __str__(self):
        summary=f'''
        -----------------------------------------
        |    {self.__class__.__name__} object:  |
        -----------------------------------------

                Number of datapoints: {len(self)}
                Directory where the data is: {self.__root}
                train: {self.__train}
                Split: {'train' if self.__train else 'test'}  
                {f'Transforms:{self.__transform}' if self.__transform else 'No transforms'}  
                {f'Target Transforms:{self.__target_transform}' if self.__target_transform else 'No target transforms'}
        -----------------------------------------

        '''
        # return __repr__(self)
        return summary
        