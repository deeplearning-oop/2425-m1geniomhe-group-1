from tensor import Tensor

class Parameter(Tensor):
    
    def __init__(self, data):
        super().__init__(data, requires_grad=True, is_leaf=True)
        
    def __repr__(self):
        return f"Parameter(data={self.data})"