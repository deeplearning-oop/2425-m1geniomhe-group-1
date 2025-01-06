from tensor import Tensor

def tensor(data, dtype=float32, requires_grad=False, is_leaf=False):
    '''
    Factory function, generates a tensor instance instead of calling the class,  
    imitates the torch.tensor() function

    ### parameters
    - data: list or numeric  
    - dtype: dtype, default float64  
    - requires_grad: bool, default False  
    - is_leaf: bool, default True  

    ### returns
    - Tensor instance

    ```
    >>> tensor([1,2,3])
    Tensor([1.0, 2.0, 3.0], requires_grad=False)
    ```
    
    '''
    return Tensor(data, dtype=dtype, requires_grad=requires_grad, is_leaf=is_leaf) 

def zeros(shape, dtype=float32, requires_grad=False, is_leaf=True):
    '''
    Factory function, generates a tensor of zeros instead of calling the class,  
    imitates the torch.zeros() function

    ### parameters
    - shape: tuple  
    - dtype: dtype, default float64  
    - requires_grad: bool, default False  
    - is_leaf: bool, default True  

    ### returns
    - Tensor instance

    ```
    >>> zeros((2,3))
    Tensor([[0., 0., 0.],
            [0., 0., 0.]], requires_grad=False)
    ```
    
    '''
    return Tensor(np.zeros(shape), dtype=dtype, requires_grad=requires_grad, is_leaf=is_leaf)

def ones(shape, dtype=float32, requires_grad=False, is_leaf=True):
    '''
    Factory function, generates a tensor of ones instead of calling the class,  
    imitates the torch.ones() function

    ### parameters
    - shape: tuple  
    - dtype: dtype, default float64  
    - requires_grad: bool, default False  
    - is_leaf: bool, default True  

    ### returns
    - Tensor instance

    ```
    >>> ones((2,3))
    Tensor([[1., 1., 1.],
            [1., 1., 1.]], requires_grad=False)
    ```
    
    '''
    return Tensor(np.ones(shape), requires_grad, is_leaf, dtype=dtype)

def randn(shape, dtype=float32, requires_grad=False, is_leaf=True):
    '''
    Factory function, generates a tensor of random numbers instead of calling the class,  
    imitates the torch.randn() function

    ### parameters
    - shape: tuple  
    - dtype: dtype, default float64  
    - requires_grad: bool, default False  
    - is_leaf: bool, default True  

    ### returns
    - Tensor instance

    ```
    >>> randn((2,3))
    Tensor([[0.1, -0.2, 0.3],
            [0.4, -0.5, 0.6]], requires_grad=False)
    ```
    
    '''
    return Tensor(np.random.randn(*shape), dtype=dtype, requires_grad=requires_grad, is_leaf=is_leaf)

def tensor_like(tensor, data, requires_grad=False, is_leaf=False):
    '''
    Factory function, generates a tensor with the same shape as another tensor instead of calling the class,  
    imitates the torch.tensor() function

    ### parameters
    - tensor: Tensor instance  
    - data: list or numeric  
    - requires_grad: bool, default False  
    - is_leaf: bool, default True  

    ### returns
    - Tensor instance

    ```
    >>> a=tensor([1,2,3])
    >>> tensor_like(a, [4,5,6])
    Tensor([4.0, 5.0, 6.0], requires_grad=False)
    ```
    
    '''
    return Tensor(data, requires_grad, is_leaf, dtype=tensor.dtype)
 
def transpose(tensor):
    '''
    generates a transposed tensor instead of calling the class method
    '''
    return tensor.T()