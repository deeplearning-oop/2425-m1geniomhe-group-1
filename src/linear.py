import numpy as np
from module import Module
from parameter import Parameter

class Linear(Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.weights = Parameter(self._random_init(input_dim, output_dim))
        self.bias = Parameter(self._random_init(output_dim))

    # def __setattr__(self, name, value):
    #         object.__setattr__(self,name, value)

    def _random_init(self, *shape):
        return np.random.randn(*shape)
    
    def forward(self, x):
        return x @ self.weights + self.bias
    
    def __repr__(self):
        return f"{self.__class__.__name__}({self.weights}, {self.bias})"