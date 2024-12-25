from enum import Enum #for dtype


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
        raise ValueError('all elements in the list must be numeric')
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

#################################################################################

class dtype(Enum):
    int64 = "int64"
    float64 = "float64"

    def __repr__(self):
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
    def __init__(self, data, dtype=float64, requires_grad=False, is_leaf=True):
        self.__data = data
        self.__dtype = dtype
        self.__requires_grad = requires_grad
        self.__is_leaf = is_leaf
        self.__dim = None #this will be set in the __setattr__ method
        

    # ----- validating attributes -----

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
            if dt not in dtype.__members__.keys():
                raise ValueError(f"Invalid dtype given: {dtype}; Valid dtypes are from {list(dtype.__members__.keys())}")
            return dtype.__members__[dt]
        except ValueError as e:
            print(f"ValueError: {e}")
            return None

    def validate_tensor_input(input_data):
        '''
        ### parameters
        - input_data: any  
        ### returns
        - dimensions: list of integers

        >>> This function handles a propagated error through a try-catch block and prints the error message

        The input should be either a numeric or a nested list of numerics (allow for numpy in later versions)  
        When validating the things to check for (in order) are:  

        1. if the input is a  non-empty list (could be nested) (or numeric): raise ValueError if not -> check_dlist(input_data)    
        2. dimensions of the list (uniformity): raise valueError if not uniform -> infer_dimensions(input_data)  

        in 1 we are checking for (when non numeric):   
        * a. top level is a list      
        * b. non-empty list (nor containing empty lists)       
        * c. lowest level is numeric    
        '''
        try:
            check_dlist(input_data)
            dimensions=infer_dimensions(input_data)
            return dimensions #or maybe just assign them in the class
        except ValueError as e:
            print("ValueError: inputData", e)
    
    def cast_dtype(self):
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
        if isinstance(self.data, list):
            return [self.cast_dtype(sublist, self.dtype) for sublist in self.data]
        return self.dtype(self.data)
    
    def __setattr__(self, name, value):
        if name == 'dtype':
            self.dtype = self.validate_dtype(value)
        if name == 'data':
            self.dim = self.validate_tensor_input(value)
            self.data= self.cast_dtype()
        super().__setattr__(name, value)