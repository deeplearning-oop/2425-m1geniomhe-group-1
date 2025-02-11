__doc__='''
-------------------
Module Module (actual name :p)
-------------------

This module contains the Module class which is the base class for all the layers and activation functions and loss and user defined models in the neural network architecture
'''

from abc import ABC, abstractmethod
from ann.nn.parameter import Parameter

class Module(ABC):
    
    def __init__(self):
        self._subclasses={} #dictionary to save all the initialized instances of the classes that inherit Module like the linear layers and the activation functions 
        self._parameters={} #dictionary to register the parameters (weights and bias) of each layer (Linear) which is a subclass of Module
        self.mode = 'train'
        
    def __setattr__(self, name, value): #to automatically register the subclasses (layers and activation functions) and parameters when they are assigned
        if isinstance(value, Module):
            self._subclasses[name]=value
        if isinstance(value, Parameter): 
            self._parameters[name]=value
        super().__setattr__(name, value)
        
        
    def parameters(self): #to be called through model.parameters()
        '''
        Returns a list of all the parameters of all the layers of the model
        '''
        parameters_list=[]
        for parameter in self._parameters.values():
            parameters_list.append(parameter)
        for layer in self._subclasses.values(): #go through each layer and add its parameters
            parameters_list.extend(layer.parameters())
        return parameters_list

    def train(self):
        self.mode = 'train'
        
    def eval(self):
        self.mode = 'eval'

    @abstractmethod
    def forward(self, x): #to be implemented by the subclasses (Model, Linear, Activation)
        pass
    
    
    def __call__(self, x): #modified to return the forward method implemented by the subclass
        return self.forward(x)
    
    
    def __repr__(self): #to visualize the neural network architechture 
        return f"{self.__class__.__name__}({self._subclasses})"   
