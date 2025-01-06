'''
----------------------------------


'''
import numpy as np
import cv2
import pandas as pd
from pathlib import Path
import gzip
import requests
from IPython.display import display

from ..tensor import Tensor

# ---- image processing ----

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

def viz_ndarray(ndarray, label=None, squeeze=False):
    '''
    takes a multidimensional array of an image and plots it, if label provided makes it a title

    params:  
    * ndarray: np.ndarray (or tensor)
    * label: str (optional)  
    * squeeze: bool (optional), if True it squeezes a 2D image thats (1, 28, 28) to (28, 28) for instance

    returns: None
    '''
    if type(ndarray)==Tensor:
        ndarray=ndarray.data #getting data as tensor

    if squeeze:
        ndarray=np.squeeze(ndarray)

    plt.imshow(ndarray, cmap='gray')
    plt.xticks([])
    plt.yticks([])
    if label:
        plt.title(f'label: {label}')
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
        print(f' >>> {filepath} already exists <<<')
    else:
        try:
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                with open(filepath, "wb") as f:
                    f.write(response.content)
                print(f" >>> File downloaded successfully as '{filepath}'.")
            else:
                print(f" >>> Failed to download file. Status code: {response.status_code}")
        except Exception as e:
            print(f"An error occurred: {e}")

    filename_no_gz=filename.replace('.gz','')
    filepath_no_gz=path/filename_no_gz


    if filepath_no_gz.exists():
        print(f' >>> {filepath_no_gz} already exists <<<')
    else:
        with open(filepath, 'rb') as f:
            file_content = f.read()
            gunzip_content = gzip.decompress(file_content)
            with open(filepath_no_gz, 'wb') as f:
                f.write(gunzip_content)

def read_idx(file_path):
    """
    reads an IDX file and returns the data as a numpy array
    
    param:
        file_path (str): Path to the IDX file
    
    returns:
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

def beautify_repr(obj):    

    def dictify_dataset(dataset):
        '''
        takes a dataset object and returns a dictionary of its attributes

        '''
        data=dataset.__dict__
        # -- if one of them is nd_array or tensor, we'll convert it to a list
        for key,val in data.items():
            if isinstance(val, np.ndarray):
                data[key]=val.tolist()
            if isinstance(val, Tensor):
                data[key]=val.data.tolist()
        # removes the private attributes
        data={key:val for key,val in data.items() if not key.startswith('_')}
        return data

    def dfy_data(data):
        '''
        takes a dictionary of data and returns a pandas dataframe

        '''
        df=pd.DataFrame(data)
        return df.T

    data=dictify_dataset(obj)
    df=dfy_data(data)
    display(df)