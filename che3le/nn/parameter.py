__doc__='''
Parameter module housing the Parameter class which is used to store the parameters of the model (W and b) and is a child of the Tensor class
'''

from che3le.tensor import Tensor

class Parameter(Tensor):
    
    def __init__(self, data):
        super().__init__(data, requires_grad=True)
        
    def __repr__(self):
        return f"Parameter(data={self.data})"