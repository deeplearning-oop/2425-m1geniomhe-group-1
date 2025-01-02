'''
Tensor module
--------------

temporary module to build a a Tensor class from a numpy array base to facilitate the mathemtaical operations
'''

from enum import Enum
import numpy as np

list_to_ndarray= lambda l: np.array(l)
ndarray_to_list= lambda a: a.tolist()


class dtype(Enum):
    '''
    Enum stype
    -----------

    might consider putting this in __init__ if importing it there is not enough
    
    :D successful test
    '''
    float32= 'float32' #might consider naming them "torch_wannabe_lib_name.float32" to make it like torch
    float64= 'float64'
    int32= 'int32'
    int64= 'int64'

    def __repr__(self):
        return f'{self.__module__.__name__}.{self.name}' #check what modules really gives
    
    def __str__(self):
        return self.value
    
    def __call__(self, x ):
        '''
        make it callable, uses:  
        ```
        >>> print(dtype.float32(3))  
        3.0
        ```
        '''
        possible_castings= {
            'float32': np.float32, #the value is a function, so we call it with the value
            'float64': np.float64,
            'int32': np.int32,
            'int64': np.int64
        }
        return possible_castings[self.value](x)
    
# -- aliasing

float32= dtype.float32
float64= dtype.float64
int32= dtype.int32
int64= dtype.int64

############################################################################################################

class Tensor:  
    def __init__(self, data, dtype= dtype.float32, requires_grad= False, is_leaf= True):
        '''
        Tensor class
        --------------
        '''
        self.__dtype = dtype 
        self.__data = data
        self.__requires_grad = requires_grad
        self.__is_leaf = is_leaf

        self.__grad = 0.0 if requires_grad else None #!!!! NEED TO HANDLE THIS ASSIGNMENT IN __setattr__ (not setter)
        self.__grad_fn = None  # To store the function for backpropagation
        # will only provide getters for now

        # self.__shape = None #assigned in __setattr__
        # self.__ndim= None #assigned in __setattr__

        # --- some helpers when setting attributes: ---
        @staticmethod
        def cast_dtype(data, data_type: dtype=dtype.float64):
            '''
            changes the dtype of the data to the specified dtype (from the dtupe class)

            :param data: the data to be casted (numeric, list, or numpy array)
            :param data_type: the dtype to cast the data to
            :return: the data casted to the specified dtype (ndarray)

            :D successful test
            ```
            >>> Tensor.cast_dtype([1,2,3], dtype.int64)
            array([1, 2, 3])
            ```
            '''
            return np.array(data, dtype= data_type.value)
        
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
            >>> validate_dtype('int16')
            ValueError: Invalid dtype given: int16; Valid dtypes are from ['int64', 'float64'] #so far, under development
            ```

            The error is handeled through try and except, so it will not raise an error, but rather return the default dtype
            '''
            try:
                dt=str(dt)
                if dt not in list(dtype.__members__.keys()):
                    print(f'testing: dt given to validte_dtype: {dt}, type: {type(dt)}')
                    raise ValueError(f"Invalid dtype given: {dtype}; Valid dtypes are from {list(dtype.__members__.keys())}")
                return dtype.__members__[dt]
            except ValueError as e:
                print(f"ValueError: {e}")
                return dtype.float64 #default dtype instead of just raising an error
        
        # ----------------------------------------------

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
            self.__data = Tensor.cast_dtype(self.__data, value)  
            # /!\ this way we can change the dtype of the tensor and the data will be casted to the new dtype automatically
        
        @dtype.deleter
        def dtype(self):
            # doesnt allow to delete the dtype, rather sets to default
            self.__dtype = dtype.float64 #default dtype better be float

        @property
        def requires_grad(self):
            return self.__requires_grad
        
        @requires_grad.setter
        def requires_grad(self, value):
            try:
                if not isinstance(value, bool):
                    raise ValueError("requires_grad must be a boolean")
                else:
                    self.__requires_grad = value
            except ValueError as e:
                print(f"ValueError: requires_grad.setter, {e}")
                # -- set to default
                self.__requires_grad = False
        @requires_grad.deleter
        def requires_grad(self):
            # doesnt allow to delete the requires_grad, rather sets to default
            self.__requires_grad = False

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
                # -- set to default
                self.__is_leaf = True
        @is_leaf.deleter
        def is_leaf(self):
            # doesnt allow to delete the is_leaf, rather sets to default
            self.__is_leaf = True

        @property
        def shape(self):
            return self.__data.__shape
        @shape.setter
        def shape(self, value):
            '''
            reshaping using numpy functionality
            
            one of the methods that is easier and more maintainable to use numpy for
            '''
            self.__data= self.__data.reshape(value)

        @property
        def ndim(self):
            return self.__data.__ndim    
        @ndim.setter  
        def ndim(self, value):
            '''
            does not allow to change the ndim
            '''
            pass

        @property
        def grad(self):
            return self.__grad
        @grad.setter
        def grad(self, value):
            '''
            ???????
            '''
            self.__grad = value

        @property  
        def grad_fn(self):
            return self.__grad_fn

        @grad_fn.setter
        def grad_fn(self, value):
            '''
            ???????
            '''
            self.__grad_fn = value

        def __setattr__(self,name,value):
            '''
            sets dtype, data and performs casting if necssary (essential to put for an initialization)
            sets shape and ndim here as they're not part of the __init__ method
            ''' 
            if name == '_Tensor__dtype':
                valid_datatype_value = Tensor.validate_dtype(value)
                self.__dict__[name] = valid_datatype_value  
                # -- CAN NOT CAST DATA HERE
                # in an  initialization step, dtype is set first to allow for validating it before assigning teh data
                # for that, data here is not present as an attribute yet for a newly instantiated object
                # this is why this will handeled in the setter of dtype in case changed, and in __setattr__ for data not to miss any possible case
            elif name == '_Tensor__data':
                # -- we'll rely on numpy for handling the data input instead of the prev defined static functiosn in 1.0 of this library
                self.__dict__[name] = np.array(value, dtype= self.__dtype.value) # no need to cast explicitly
                # /!\ SETTING SHAPE AND NDIM /!\
                self.__shape = self.__data.shape
                self.__ndim = self.__data.ndim

            elif name == '_Tensor__shape':
                # -- this will be handled in the setter of shape but imp to assign ndim when shape is assigned
                self.__dict__[name] = value
                self.__dict__['_Tensor__ndim'] = len(value)  

            else:
                self.__dict__[name] = value

        def __repr__(self):
            return f"Tensor({self.__data}, dtype={self.__dtype}, requires_grad={self.__requires_grad}, is_leaf={self.__is_leaf})"  
        def __str__(self):
            return f"Tensor({self.__data})"
        

        # -- some magic methods that benefit from numpy's magic methods
        def __len__(self):
            return len(self.__data)
        def __getitem__(self, index):
            '''
            __getitem__ returns each item as a Tensor object
            '''
            temp= Tensor(self.__data[index], dtype= self.__dtype, requires_grad= self.__requires_grad, is_leaf= self.__is_leaf)
            return temp
        def __setitem__(self, index, value):
            self.__data[index] = value
        def __iter__(self):
            '''
            __iter__ has to return each value as a Tensor object
            '''
            for i in self.__data:
                yield Tensor(i, dtype= self.__dtype, requires_grad= self.__requires_grad, is_leaf= self.__is_leaf)

        # -- mathematical operations, USES COMPUTATIONAL GRAPH IDEA
        def __add__(self, other):
            if not isinstance(other, Tensor):
                other= Tensor(other) # makes it default because dtypes might not match for instance

            result= Tensor(self.__data + other.data, dtype= self.__dtype, requires_grad= self.__requires_grad or other.requires_grad)

            if result.__requires_grad:
                result.__is_leaf= False

                def grad_fn(grad):
                    '''
                    the gradient function for the addition operation
                    ----------------
                    takes the gradient of the result and returns a tuple of the gradients of the operands
                    '''
                    return grad, grad 
                
                result.__grad_fn= grad_fn
                print(f' -- testing addition grad_fn: {result.__grad_fn}')
            return result
        
        # -- check this for a graph implementation --
        # def __sub__(self, other):
        #     if not isinstance(other, Tensor):
        #         other = Tensor(other)

        #     result = Tensor(self.data - other.data, requires_grad=self.requires_grad or other.requires_grad)
        #     result.is_leaf = False
        #     result.parents = [self, other]

        #     def grad_fn(grad):
        #         grad_self = grad
        #         grad_other = -grad
        #         return grad_self, grad_other

        #     result.grad_fn = (grad_fn, self, other)
        #     return result
        
        def __sub__(self, other):
            '''
            same as addition, but for subtraction
            '''
            if not isinstance(other, Tensor):
                other= Tensor(other)

            result= Tensor(self.__data - other.data, dtype= self.__dtype, requires_grad= self.__requires_grad or other.requires_grad)

            if result.__requires_grad:
                result.__is_leaf= False

                def grad_fn(grad):
                    '''
                    the gradient function for the subtraction operation
                    ----------------
                    takes the gradient of the result and returns a tuple of the gradients of the operands
                    '''
                    return grad, -grad 
                
                result.__grad_fn= grad_fn
                print(f' -- testing subtraction grad_fn: {result.__grad_fn}')

            return result
        
        def __mul__(self, other):
            '''
            same as addition, but for multiplication
            '''
            if not isinstance(other, Tensor):
                other= Tensor(other)

            result= Tensor(self.__data * other.data, dtype= self.__dtype, requires_grad= self.__requires_grad or other.requires_grad)

            if result.__requires_grad:
                result.__is_leaf= False

                def grad_fn(grad):
                    '''
                    the gradient function for the multiplication operation
                    ----------------
                    takes the gradient of the result and returns a tuple of the gradients of the operands
                    '''
                    return grad*other.data, grad*self.__data
                
                result.__grad_fn= grad_fn
                print(f' -- testing multiplication grad_fn: {result.__grad_fn}')

            return result
        
        def __truediv__(self, other):
            '''
            same as addition, but for division
            '''
            if not isinstance(other, Tensor):
                other= Tensor(other)

            result= Tensor(self.__data / other.data, dtype= self.__dtype, requires_grad= self.__requires_grad or other.requires_grad)

            if result.__requires_grad:
                result.__is_leaf= False

                def grad_fn(grad):
                    '''
                    the gradient function for the division operation
                    ----------------
                    takes the gradient of the result and returns a tuple of the gradients of the operands
                    '''
                    return grad/other.data, -grad*self.__data/(other.data**2)
                
                result.__grad_fn= grad_fn
                print(f' -- testing division grad_fn: {result.__grad_fn}')

            return result
        
        def __pow__(self, other):
            '''
            for power now

            the main difference is in the way the gradient is calculated (derivative of x^a is a*x^(a-1))
            '''
            if not isinstance(other, Tensor):
                other= Tensor(other)

            result= Tensor(self.__data ** other.data, dtype= self.__dtype, requires_grad= self.__requires_grad or other.requires_grad)

            if result.__requires_grad:
                result.__is_leaf= False

                def grad_fn(grad):
                    '''
                    the gradient function for the power operation
                    ----------------
                    takes the gradient of the result and returns a tuple of the gradients of the operands
                    '''
                    return grad*other.data*(self.__data**(other.data-1)), grad*(self.__data**other.data)*np.log(self.__data)
                
                result.__grad_fn= grad_fn
                print(f' -- testing power grad_fn: {result.__grad_fn}')

            return result
        
        def __neg__(self):
            '''
            for negation (unary operation, so no need for the other operand)
            '''
            result= Tensor(-self.__data, dtype= self.__dtype, requires_grad= self.__requires_grad)

            if result.__requires_grad:
                result.__is_leaf= False

                def grad_fn(grad):
                    '''
                    the gradient function for the negation operation
                    ----------------
                    takes the gradient of the result and returns a tuple of the gradients of the operands
                    '''
                    return -grad
                
                result.__grad_fn= grad_fn
                print(f' -- testing negation grad_fn: {result.__grad_fn}')

            return result
                
        def __abs__(self):
            '''
            for absolute value
            '''
            result= Tensor(abs(self.__data), dtype= self.__dtype, requires_grad= self.__requires_grad)

            if result.__requires_grad:
                result.__is_leaf= False

                def grad_fn(grad):
                    '''
                    the gradient function for the absolute value operation
                    ----------------
                    takes the gradient of the result and returns a tuple of the gradients of the operands
                    '''
                    return grad*np.sign(self.__data)
                
                result.__grad_fn= grad_fn
                print(f' -- testing absolute value grad_fn: {result.__grad_fn}')

            return result
        
        def __matmul__(self, other):
            '''
            /!\ this is for matrix multiplication, not element-wise multiplication  
            
            this can be done through the @ operator in python
            '''
            if not isinstance(other, Tensor):
                other= Tensor(other)

            result= Tensor(self.__data @ other.data, dtype= self.__dtype, requires_grad= self.__requires_grad or other.requires_grad)

            if result.__requires_grad:
                result.__is_leaf= False

                def grad_fn(grad):
                    '''
                    the gradient function for the matrix multiplication operation
                    ----------------
                    takes the gradient of the result and returns a tuple of the gradients of the operands
                    '''
                    return grad @ other.data.T, self.__data.T @ grad
                    # by definition of matrix multiplication, the gradient of the result w.r.t. the first operand is the result of the multiplication of the gradient w.r.t. the result and the transpose of the second operand
                
                result.__grad_fn= grad_fn
                print(f' -- testing matrix multiplication grad_fn: {result.__grad_fn}')


            return result
        
    def backward(self, grad=1.0):
        '''
        this method will be used to backpropagate the gradient through the computational graph  

        ### params:
        - grad: the gradient to backpropagate (default is 1.0, because the gradient of the loss w.r.t. itself is 1.0)  

        ### returns:
        - None

        /!\ errors are not handeled yet
        '''

        if not self.__requires_grad:
            raise ValueError("requires_grad is False, so no gradient to backpropagate")
        
        if self.grad is None:
            self.__grad = grad
        else:
            self.__grad += grad

        if self.__grad_fn:
            # ????  
            # for a non-leaf node, propagate the gradient through the graph  
            grad_fn, *parents = self.grad_fn
            parent_grads = grad_fn(grad)

            for parent, parent_grad in zip(parents, parent_grads):
                parent.backward(parent_grad)   


      