from enum import Enum #for dtype
import numpy as np #for matrix operations


#################################################################################
# --------------- helper functions to validate input data ---------------
# --------------- will add this to a utils or helper.py file and import it ---------------


def is_emptylist(l):
    '''
    returns True if l has at the lowest level at least one empty list
    
    e.g.

    ```
    >>> is_emptylist([])
    True
    >>> is_emptylist([1])
    False
    >>> is_emptylist(1)  
    False
    >>> is_emptylist([[]])
    True
    >>> is_emptylist([[],[]])  
    True
    >>> is_emptylist([[],[1]])
    True
    >>> is_emptylist([[1,2],[1,2]])
    False
    ```
    '''
    if isinstance(l, list):
        if not l:  
            return True
        return any(is_emptylist(x) for x in l)  #recursive call
    return False

def is_numeric(x):
    '''
    takes x and returns True if x is a numeric type 
    
    e.g.

    ```
    >>> is_numeric(1)
    True
    >>> is_numeric(1.0)
    True
    >>> is_numeric('a')
    False
    >>> is_numeric([1])
    False
    ```
    '''
    acceptable_numeric_types = (int, float, np.int64, np.float64, np.int32, np.float32)
    for i in acceptable_numeric_types:
        if isinstance(x, i):
            return True
    return False

def is_inner_numeric(l:list):
    '''
    recursive function that takes a list l and returns True if all the inner elements of l are numeric (depends on is_numeric())

    e.g.

    ```
    >>> is_inner_numeric([1,2,3])
    True
    >>> is_inner_numeric([1,2,'a'])
    False
    >>> is_inner_numeric([1,[2,3]])
    True
    >>> is_inner_numeric([1,[2,'a']])
    False
    ```
    '''
    if isinstance(l,list):
        return all(is_inner_numeric(x) for x in l)
    else:
        return is_numeric(l)
    
def check_dlist(l):
    '''
    takes any input type and reurns either True or raises a ValueError  
    makes sure the input is either a numeric or a non empty dlist of numerics, depends on is_inner_numeric() and is_numeric() and is_emptylist()

    THROW ValueError if the input is not a numeric or a dlist of numerics
    
    '''
    if is_emptylist(l):
        raise ValueError('empty list provided')
    if not is_inner_numeric(l):
        raise ValueError('all elements in the LIST must be NUMERIC')
    return True

def infer_dimensions(nested_list):
    '''
    recursively infer the dimensions of a nested list and validate uniformity  

    THROW ERROR IF NOT UNIFORM

    ```
    >>> infer_dimensions([1,2,3]) #vector
    [3]
    >>> infer_dimensions([1]) #vector
    [1]
    >>> infer_dimensions(1)  #scalar
    []  
    >>> infer_dimensions([[1,2],[3,4]]) #matrix
    [2, 2]  
    >>> infer_dimensions([[[1,2],[3,4]],[[5,6],[7,8]]]) #3D tensor
    [2, 2, 2]  
    >>> infer_dimensions([[[1,2],[3,4]],[[5,6],[7]]]) 
    ValueError: dimension mismatch detected: [[2, 2], [2, 1]]
    '''
    if isinstance(nested_list, list):
        if len(nested_list) == 0: #if empty inner list = dimension 0
            'base case for scalars, reurn dim 0'
            return [0]  
        sub_shapes = [infer_dimensions(sublist) for sublist in nested_list]
        
        if len(set(map(tuple, sub_shapes))) > 1:  
            '''
            # this condition takes all shapes of lists at the same level which are lists inside sub_shapes
            # makes them tuples and remove duplicates (set())
            # length should be 1 if the lists are uniform in shape
            '''

            raise ValueError(f"dimension mismatch detected: {sub_shapes}")
    
        return [len(nested_list)] + sub_shapes[0]  #combine this level with sub-dimensions so this way we have [2,2] for [[1,2],[3,4]] (sub_shapes[0] is the only item in the list)
    
    #if not a list (a scalar), no dimensions, need to return a list of length 0 in order to check for at the next base case
    return [] 


# ---------------- helper functions for math operations ----------------
# def add_nested_lists(list1, list2):
#     '''
#     this performs addition on multidimensional lists (including numerics)
    
#     *might ditch this and move to numpy addition if performance is an issue
#     '''
#     if isinstance(list1, list) and isinstance(list2, list):
#         return [add_nested_lists(x, y) for x, y in zip(list1, list2)]
#     else:
#         return list1 + list2
    
# def subtract_nested_lists(list1, list2):
#     '''this performs subtraction on multidimensional lists (including numerics)'''
#     if isinstance(list1, list) and isinstance(list2, list):
#         return [subtract_nested_lists(x, y) for x, y in zip(list1, list2)]
#     else:
#         return list1 - list2

# def scalar_multiply(l:list, scalar):
#     '''
#     WONT USE IT ANYMORE WILL USE NUMPY OPS
#     a function to multiply a list by a scalar (2 or more dimensions) 

#     ### params:
#         * l: list to multiply
#         * scalar: scalar to multiply by
#     ### returns:
#         * multiplied list

#     it does this operation through an ndarray intermediary

#     this will be used in the * method of the tensor class, making these 2 ops equivalent:  

#     ```
#     >>> scalar_multiply([[1,2],[3,4]], 2)
#     [[2, 4], [6, 8]]
#     >>> t=tensor([[1,2],[3,4]])
#     >>> t*2
#     [[2, 4], [6, 8]]
#     ```

#     '''
#     ndarray=list_to_ndarray(l)
#     multiplied=ndarray*scalar
#     return ndarray_to_list(multiplied)

def fix_dtype(my_new_instance, ref_instance):
    '''this function is made to fix the dtype of the new instance to be the same as the reference instance IFF the reference instance is of type float64'''
    if ref_instance.dtype==float64: #access it without __ thanks to setters
        my_new_instance.dtype=float64
    
list_to_ndarray= lambda l: np.array(l)
ndarray_to_list= lambda a: a.tolist()

tensor_to_ndarray= lambda t: np.array(t.data)  
ndarray_to_tensor= lambda a: Tensor(a.tolist())
    
def transpose(l:list):
    '''
    a function to transpose a list (2 or more dimensions) 

    ### params:
        * l: list to transpose
    ### returns:
        * transposed list

    it does this operation through an ndarray intermediary

    this will be used in the T method of the tensor class, making these 2 ops equivalent:  

    ```
    >>> transpose([[1,2],[3,4]])
    [[1, 3], [2, 4]]
    >>> t=tensor([[1,2],[3,4]])
    >>> t.T
    [[1, 3], [2, 4]]
    ```

    '''
    ndarray=list_to_ndarray(l)
    transposed=ndarray.T
    return ndarray_to_list(transposed)



def element_wise_multiply(l1, l2):
    '''
    a function to multiply 2 lists, element-wise

    ### params:
        * l1: list 1
        * l2: list 2
    ### returns:
        * multiplied list

    it does this operation through an ndarray intermediary

    this will be used in the * method of the tensor class (__mul__), making these 2 ops equivalent:  

    ```
    >>> multiply([[1,2],[3,4]], [[1,2],[3,4]])
    [[1, 4], [9, 16]]
    >>> t1=tensor([[1,2],[3,4]])
    >>> t2=tensor([[1,2],[3,4]])
    >>> t1*t2
    [[1, 4], [9, 16]]
    ```

    '''
    
    ndarray1=list_to_ndarray(l1)
    ndarray2=list_to_ndarray(l2)
    multiplied=ndarray1*ndarray2
    return ndarray_to_list(multiplied)

def matrix_multiply(l1, l2):
    '''
    a function to multiply 2 matrices

    ### params:
        * l1: matrix 1
        * l2: matrix 2
    ### returns:
        * multiplied matrix

    it does this operation through an ndarray intermediary

    this will be used in the @ method of the tensor class (__matmul__), making these 2 ops equivalent:  

    ```
    >>> matrix_multiply([[1,2],[3,4]], [[1,2],[3,4]])
    [[7, 10], [15, 22]]
    >>> t1=tensor([[1,2],[3,4]])
    >>> t2=tensor([[1,2],[3,4]])
    >>> t1@t2
    [[7, 10], [15, 22]]
    ```

    '''
    
    ndarray1=list_to_ndarray(l1)
    ndarray2=list_to_ndarray(l2)
    multiplied=ndarray1@ndarray2
    return ndarray_to_list(multiplied)

dot=matrix_multiply #aliasing

def addition(l1, l2):  
    '''
    a function to add 2 multidimensional lists even if numeric

    ### params:
        * l1: list 1
        * l2: list 2  

    ### returns:
        * added list
    
    if both are numeric, returns the sum of the 2 numbers  \ 
    it does an automatic conversion to numpy arrays and back to lists to perform the addition  \ 
    no need to check for uniformity of dimensions, numpy will handle it  

    this will be used in the + method of the tensor class (__add__) eventually\ 
    AIMING TO USE THIS WHILE DEVELOPING THE COMPUTATIONAL GRAPH

    '''
    ndarray1=list_to_ndarray(l1)
    ndarray2=list_to_ndarray(l2)
    added=ndarray1+ndarray2
    return ndarray_to_list(added)


def substraction(l1, l2):
    '''
    a function to subtract 2 multidimensional lists even if numeric

    ### params:
        * l1: list 1
        * l2: list 2  

    ### returns:
        * subtracted list
    
    if both are numeric, returns the difference of the 2 numbers  \ 
    it does an automatic conversion to numpy arrays and back to lists to perform the subtraction  \ 
    no need to check for uniformity of dimensions, numpy will handle it  

    this will be used in the - method of the tensor class (__sub__) eventually\ 
    AIMING TO USE THIS WHILE DEVELOPING THE COMPUTATIONAL GRAPH

    '''
    ndarray1=list_to_ndarray(l1)
    ndarray2=list_to_ndarray(l2)
    subtracted=ndarray1-ndarray2
    return ndarray_to_list(subtracted)

def true_division(l1, l2):
    '''
    a function to divide 2 multidimensional lists even if numeric

    ### params:
        * l1: list 1
        * l2: list 2  

    ### returns:
        * divided list
    
    if both are numeric, returns the division of the 2 numbers  \ 
    it does an automatic conversion to numpy arrays and back to lists to perform the division  \ 
    no need to check for uniformity of dimensions, numpy will handle it  

    this will be used in the / method of the tensor class (__truediv__) eventually\ 
    AIMING TO USE THIS WHILE DEVELOPING THE COMPUTATIONAL GRAPH

    '''
    ndarray1=list_to_ndarray(l1)
    ndarray2=list_to_ndarray(l2)
    divided=ndarray1/ndarray2
    return ndarray_to_list(divided)

def random_mdlist(shape):
    '''
    a function to generate a random multidimensional list of a given shape

    ### params:
        * shape: list of integers, shape of the list to generate  

    ### returns:
        * random list of the given shape
    
    this will be used to generate random tensors for testing purposes

    '''
    return np.random.rand(*shape).tolist()

#################################################################################

class dtype(Enum):
    int64 = "int64"
    float64 = "float64"

    def __repr__(self):
        return self.value

    def __str__(self):
        #retuns "int64" or "float64"
        return self.value

    def __call__(self, x):
        '''make if callable, uses:
        ```
        >>> dtype.int64(1.7)
        1
        >>> dtype.float64(1)
        1.0
        '''
        if self == dtype.int64:
            return int(x)
        elif self == dtype.float64:
            return float(x)
        else:
            print(f"Unknown dtype: {self}")

# -- aliasing
int64 = dtype.int64
float64 = dtype.float64

class Tensor:
    def __init__(self, data, dtype='float64', requires_grad=False, is_leaf=True):
        self.__dtype = dtype #important to declare it before data because we'll use to convert the data type
        self.__data = data
        self.__requires_grad = requires_grad
        self.__is_leaf = is_leaf

        self.__grad = 0.0 if requires_grad else None #!!!! NEED TO HANDLE THIS ASSIGNMENT IN __setattr__ (not setter)
        self.__grad_fn = None  # To store the function for backpropagation
        # will only provide getters for now

        # self.__shape = None #assigned in __setattr__
        # self.__ndim= None #assigned in __setattr__

    @property
    def data(self):
        return self.__data
    @data.setter
    def data(self, value):
        self.__data = value
    
    @property
    def dtype(self):
        return self.__dtype
    @dtype.setter
    def dtype(self, value):
        self.__dtype = value
        self.__data=Tensor.cast_dtype(self.__data,  self.__dtype)
        # /!\ this way we can change the dtype of the tensor and the data will be casted to the new dtype automatically
    
    @dtype.deleter
    def dtype(self): #deletes dtype by setting it to default
        self.__dtype=float64
    
    @property
    def requires_grad(self):
        return self.__requires_grad
    @requires_grad.setter
    def requires_grad(self, value):
        # print('test, requires_grad setter is running')
        try:
            if not isinstance(value, bool):
                raise ValueError("requires_grad must be a boolean")
            else:
                self.__requires_grad = value
        except ValueError as e:
            print(f"ValueError: requires_grad.setter, {e}") 
        
    @requires_grad.deleter
    def requires_grad(self): #deletes requires_grad by setting it to default
        self.__requires_grad=False

    @property
    def is_leaf(self):
        return self.__is_leaf
    @is_leaf.setter
    def is_leaf(self, value):
        try:
            if not isinstance(value, bool):
                raise ValueError("is_leaf must be a boolean")
            else:
                self.__is_leaf = value
        except ValueError as e:
            print(f"ValueError: is_leaf.setter, {e}")
        
    @is_leaf.deleter
    def is_leaf(self): #deletes is_leaf by setting it to default
        self.__is_leaf=True

    @property
    def shape(self):
        return self.__shape
    @shape.setter
    def shape(self):
        self.__shape = infer_dimensions(self.__data)

    @property
    def ndim(self):
        return self.__ndim
    @ndim.setter
    def ndim(self):
        self.__ndim = len(self.__shape)


    @property
    def grad(self):
        return self.__grad
    
    @property
    def grad_fn(self):
        return self.__grad_fn
        

    # ----- validating attributes -----

    #should i use .data or .__data here !!!!!!!!???????
    @staticmethod
    def validate_dtype(dt):
        '''
        ### parameters
        - dt: str  
        ### returns  
        - dtype object
        check if dtype is within the dtype enumerate

        ```
        >>> validate_dtype('int64')
        >>> validate_dtype('int32')
        ValueError: Invalid dtype given: int32; Valid dtypes are from ['int64', 'float64'] #so far, under development
        ```
        '''
        try:
            dt=str(dt)
            if dt not in list(dtype.__members__.keys()):
                print(f'testing: dt given to validte_dtype: {dt}, type: {type(dt)}')
                raise ValueError(f"Invalid dtype given: {dtype}; Valid dtypes are from {list(dtype.__members__.keys())}")
            return dtype.__members__[dt]
        except ValueError as e:
            print(f"ValueError: {e}")
            return None

    #should i use .data or .__data here !!!!!!!!???????
    @staticmethod
    def validate_tensor_input(input_data):
        '''
        ### parameters
        - input_data: any  
        ### returns
        - dimensions: list of integers

        >>> This function handles a propagated error through a try-except block and prints the error message

        The input should be either a numeric or a nested list of numerics OR numpy arrays.
        When validating the things to check for (in order) are:  

        1. if the input is a  non-empty list (could be nested) (or numeric): raise ValueError if not -> check_dlist(input_data)    
        2. dimensions of the list (uniformity): raise valueError if not uniform -> infer_dimensions(input_data)  

        in 1 we are checking for (when non numeric):   
        * a. top level is a list      
        * b. non-empty list (nor containing empty lists)       
        * c. lowest level is numeric    
        '''
        # -- if its a numpy array
        if isinstance(input_data, np.ndarray):
            input_data = input_data.tolist()
        try:
            check_dlist(input_data)
            dimensions=infer_dimensions(input_data)
            return dimensions #or maybe just assign them in the class
        except ValueError as e:
            print("ValueError: inputData", e)
    

    #should i use .data or .__data here !!!!!!!!???????
    @staticmethod
    def cast_dtype(l, dt=float64):
        '''
        recursively cast the elements of a nested list to a given dtype (default is float64)

        e.g. the function outside the class would be: 
        
        ```
        >>> cast_dtype([1,2,3])
        [1.0, 2.0, 3.0]
        >>> cast_dtype([1,2,3], dtype.int64)
        [1, 2, 3]
        ```
        '''
        if isinstance(l, list):
            return [Tensor.cast_dtype(sublist, dt) for sublist in l]
        return dt(l)
    
    def __setattr__(self, name, value):
        '''sets dtype, data and shape (shape and ndim will be set using the setter)'''
        # print(f'testing attributes automatic name mandling: {name}, the list is {self.__dict__}')
        # print('test')
        if name == '_Tensor__dtype':
            valid_datatype_value = Tensor.validate_dtype(value)
            self.__dict__[name] = valid_datatype_value

            # -- CAN NOT PUT THIS HERE BCS DATA IS NOT DEFINED YET WHILE INSTANCIATING FOR !ST TIME
            # self.__data=Tensor.cast_dtype(self.__data, value)

        elif name == '_Tensor__data':
            dimensions=Tensor.validate_tensor_input(value)

            self.__dict__[name] = Tensor.cast_dtype(value, self.__dtype)
            self.__dict__['_Tensor__shape'] = dimensions
            self.__dict__['_Tensor__ndim'] = len(dimensions) if type(dimensions)==list else 0

        elif name=='_Tensor__shape':
            dimensions=Tensor.validate_tensor_input(self.__data)
            self.__dict__[name] = dimensions
            self.__dict__['_Tensor__ndim'] = len(self.__shape)

        elif name=='_Tensor__ndim':
            self.__dict__[name] = len(self.__shape)

        else:
            super().__setattr__(name, value)


    def __repr__(self):
        return f"Tensor({self.__data}, dtype={self.__dtype}, requires_grad={self.__requires_grad}, is_leaf={self.__is_leaf})"
    
    def __str__(self):
        return f"Tensor({self.__data})"
        # return f"Tensor({self.__data})"  
    
    def __len__(self): 
        '''returns the length of the tensor, as is the length of the upper list'''
        try:
            if self.__shape==[]:
                raise TypeError("tensor is a scalar with 0 dimensions")
        except TypeError as e:
            print(f"TypeError: len(), {e}")
            return None
            
        return self.__shape[0]
    
    def __iter__(self):
        '''iterates over the tensor data'''
        if is_numeric(self.__data):
            yield self
        else:
            for i in self.__data:
                yield Tensor(i, self.__dtype, self.__requires_grad, self.__is_leaf)

    def __getitem__(self, key):
        '''returns the item at the given index'''
        return Tensor(self.__data[key], self.__dtype, self.__requires_grad, self.__is_leaf)
        


    def __add__(self, other):
        '''Addition
        ---------------
        this dunder method is called when using the + operator on two tensors

        ### parameters
        - other: Tensor

        ### returns
        - Tensor

        ### example
        ```
        >>> t1=tensor([1,2,3])
        >>> t2=tensor([1,2,3])
        >>> t1+t2
        Tensor([2.0, 4.0, 6.0], dtype=float64, requires_grad=False, is_leaf=False)
        ```
        
        
        '''
        if not isinstance(other, Tensor):
            other = Tensor(other)

        data_type=int64
        if self.dtype==float64 or other.dtype==float64:
            data_type=float64
        result = Tensor(addition(self.data, other.data),dtype=data_type, requires_grad=self.requires_grad or other.requires_grad)
        # fix_dtype(result, other) #if one of self or other is float, this has to be float so its fixing the dtype if needed

        if result.requires_grad:
            result.is_leaf = False

            def grad_fn(grad):
                return grad, grad

            result.grad_fn = (grad_fn, self, other)
            print("grad_fn=", grad_fn)

        return result
        
    # def __sub__(self, other):
    #     '''subtracts two tensors'''
    #     if not isinstance(other, Tensor):
    #         other = Tensor(other)

    #     data_type=int64
    #     if self.dtype==float64 or other.dtype==float64:
    #         data_type=float64
    #     result = Tensor(substraction(self.data, other.data),dtype=data_type, requires_grad=self.requires_grad or other.requires_grad)

    #     if result.requires_grad:
    #         result.is_leaf = False

    #         def grad_fn(grad):
    #             return grad, grad

    #         result.grad_fn = (grad_fn, self, other)
    #         print("grad_fn=", grad_fn)

        
        
    #     return result
    
    def __sub__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)

        data_type=int64
        if self.dtype==float64 or other.dtype==float64:
            data_type=float64
        result = Tensor(substraction(self.data, other.data), requires_grad=self.requires_grad or other.requires_grad, dtype=data_type)
        result.is_leaf = False
        result.parents = [self, other]

        def grad_fn(grad):
            grad_self = grad
            grad_other = -grad
            return grad_self, grad_other

        result.grad_fn = (grad_fn, self, other)
        return result

    # def __mul__(self, other):
    #     '''
    #     element-wise multiplication of two tensors,  
    #     this dunder method is called when using the * operator on two tensors

    #     IMPORTANT!  
    #         in case one of teh tensors has type dtype.float64, the result will be of type dtype.float64

    #     in pytorch an equivalent function is torch.mul() or self.mul(). We will allow for self.mul(other) by adding a method to the class `mul=__mul__`

    #     ### parameters
    #     - other: Tensor

    #     ### returns
    #     - Tensor
        
        
    #     '''
    #     # if self.__shape != other.__shape:
    #     #     raise ValueError(f"shape mismatch: {self.__shape} != {other.__shape}, can only multiply tensors of the same shape")

    #     #-- numpy will handle the shape mismatch
    #     new_data = element_wise_multiply(self.__data, other.__data)
    #     new_tensor= Tensor(new_data, self.__dtype, self.__requires_grad, self.__is_leaf)
    #     fix_dtype(new_tensor, other) #if other is float this will be float since it takes by default teh dtype of self
    #     return new_tensor

    def __mul__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)


        data_type=int64
        if self.dtype==float64 or other.dtype==float64:
            data_type=float64
        result = Tensor(list_to_ndarray(), requires_grad=self.requires_grad or other.requires_grad,dtype=data_type)

        if result.requires_grad:
            result.is_leaf = False

            def grad_fn(grad):
                grad_self = grad * other.data
                grad_other = grad * self.data
                return grad_self, grad_other

            result.grad_fn = (grad_fn, self, other)

        return result
    
    mut=__mul__ #aliasing
    mm=__mul__ #aliasing

    # def __matmul__(self, other):
    #     '''
    #     matrix multiplication of two tensors,  
    #     this dunder method is called when using the @ operator on two tensors

    #     IMPORTANT!  
    #         in case one of teh tensors has type dtype.float64, the result will be of type dtype.float64

    #     in pytorch an equivalent function is torch.matmul() or self.matmul(). We will allow for self.matmul(other) by adding a method to the class `matmul=__matmul__`

    #     ### parameters
    #     - other: Tensor

    #     ### returns
    #     - Tensor
        
        
    #     '''
    #     if self.__shape[-1] != other.__shape[0]:
    #         raise ValueError(f"shape mismatch: {self.__shape} != {other.__shape}, can only multiply tensors of the same shape")
    #     new_data = matrix_multiply(self.__data, other.__data)
    #     new_tensor= Tensor(new_data, self.__dtype, self.__requires_grad, self.__is_leaf)
    #     fix_dtype(new_tensor, other)
    #     return new_tensor

    def __matmul__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)

        data_type=int64
        if self.dtype==float64 or other.dtype==float64:
            data_type=float64
        result = Tensor(list_to_ndarray(self.data)@list_to_ndarray(other.data), requires_grad=self.requires_grad or other.requires_grad,dtype=data_type)

        if result.requires_grad:
            result.is_leaf = False

            def grad_fn(grad):
                grad_self = grad @ other.data.T
                grad_other = self.data.T @ grad
                return grad_self, grad_other

            result.grad_fn = (grad_fn, self, other)
    
    matmul=__matmul__ #aliasing

    def __truediv__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)

        data_type=int64
        if self.dtype==float64 or other.dtype==float64:
            data_type=float64
        result = Tensor(self.data / other.data, requires_grad=self.requires_grad or other.requires_grad, dtype=data_type)
        result.is_leaf = False
        result.parents = [self, other]

        def grad_fn(grad):
            grad_self = grad / other.data
            grad_other = -grad * self.data / (other.data ** 2)
            return grad_self, grad_other

        result.grad_fn = (grad_fn, self, other)
        return result

#####################################################################################################################
    
def tensor(data, dtype=float64, requires_grad=False, is_leaf=True):
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
    Tensor([1.0, 2.0, 3.0], dtype=float64, requires_grad=False, is_leaf=True)
    ```
    
    '''
    return Tensor(data, dtype, requires_grad, is_leaf)  

def random(shape, dtype=float64, requires_grad=False, is_leaf=True):
    '''
    Factory function, generates a random tensor instance instead of calling the class,  
    imitates the torch.randn() function

    ### parameters
    - shape: list of integers  
    - dtype: dtype, default float64  
    - requires_grad: bool, default False  
    - is_leaf: bool, default True  

    ### returns
    - Tensor instance

    ```
    >>> random([2,2])
    Tensor([[0.5488135039273248, 0.7151893663724195], [0.6027633760716439, 0.5448831829968969]], dtype=float64, requires_grad=False, is_leaf=True)
    ```
    
    '''
    return Tensor(random_mdlist(shape), dtype, requires_grad, is_leaf)