from .activation import Activation
from .linear import Linear
from .loss import Loss
from .module import Module
from .optimizer import Optimizer
from .parameter import Parameter

from ..tensor import *

# chek why these imports are not working on their own in the modules (can not import them direct;y need to do it from che3le.nn)

__all__ = ["Activation", "Linear", "Loss", "Module", "Optimizer", "Parameter"]

__doc__='''
>>> nn package <<<
-------------------
core neural network module containing all the necessary classes for building a neural network
'''
