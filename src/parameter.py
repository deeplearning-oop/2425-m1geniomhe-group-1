from tensor import Tensor

class Parameter(Tensor):
    def __init__(self, value):
        super().__init__(value, requires_grad=True)
        
    def __str__(self):
        return str(self.data)