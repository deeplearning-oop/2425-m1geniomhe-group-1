import numpy as np
from module import Module
from parameter import Parameter

class Linear(Module):
    
    def __init__(self, input_size, output_size):
        super().__init__()
        self.input_size=input_size
        self.output_size=output_size
        self.weights=Parameter(np.random.rand(output_size, input_size)-0.5) #initialize weights (as type Parameter) to random values centered around 0
        self.bias=Parameter(np.zeros((output_size,1))) #initialize bias (as type Parameter) to 0

    
    @property  
    def input_size(self):
        return self.__input_size
    
    @input_size.setter #performs some verifications
    def input_size(self, input_size):
        if not isinstance(input_size, int):
            raise ValueError("input_size must be an integer")
        if input_size <= 0:
            raise ValueError("input_size must be positive")
        self.__input_size = input_size
        
        
    @property
    def output_size(self):
        return self.__output_size
    
    @output_size.setter #performs some verifications
    def output_size(self, output_size):
        if not isinstance(output_size, int):
            raise ValueError("output_size must be an integer")
        if output_size <= 0:
            raise ValueError("output_size must be positive")
        self.__output_size = output_size
        
        
    def forward(self, x): #applies affine linear transformation
        # print(f'TROUBLSHOOTING: shapes of W {self.weights.shape} and x {x.shape} and bias {self.bias.shape}') 
        return self.weights @ x + self.bias
    
        
    def __repr__(self):
        return f"Linear(input_size={self.input_size}, output_size={self.output_size})"
    