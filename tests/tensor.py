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


    


class Tensor:
    def __init__(self, data, requires_grad=False, is_leaf=True):
        self.data = data
        self.requires_grad = requires_grad
        self.is_leaf = is_leaf
        self.dim = None

    def __setattr__(self, name, value):
        if name == 'data':
            self.dim = self.validate_tensor_input(value)
        super().__setattr__(name, value)

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

        in 1 we are checking for (when non numeric):  \ 
        a. top level is a list   \   
        b. non-empty list (nor containing empty lists)     \  
        c. lowest level is numeric    
        '''
        try:
            check_dlist(input_data)
            dimensions=infer_dimensions(input_data)
            return dimensions #or maybe just assign them in the class
        except ValueError as e:
            print("ValueError: inputData", e)
