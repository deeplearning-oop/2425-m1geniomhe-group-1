import numpy as np
from module import Module
from parameter import Parameter

class Linear(Module):
    def __init__(self, output_dim, input_dim):
        super().__init__()
        self.weights = Parameter(self._random_init(output_dim, input_dim))
        self.bias = Parameter(self._random_init(output_dim,1))

    # def __setattr__(self, name, value):
    #         object.__setattr__(self,name, value)

    def _random_init(self, *shape):
        return np.random.rand(*shape) - 0.5 #centered around 0
    
    def forward(self, x):
        return self.weights @ x + self.bias
    
    def __repr__(self):
        return f"{self.__class__.__name__}({self.weights}, {self.bias})"