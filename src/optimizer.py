import numpy as np
from tensor import Tensor

class Optimizer:
    def __init__(self, parameters, lr=0.01):
        self.parameters = [p for p in parameters]
        self.lr = lr

    # def step(self):
    #     for p in self.parameters:
    #         if p.grad is not None:
    #             p.data -= p.grad * self.lr

    def zero_grad(self):
        for p in self.parameters:
            if p.grad is not None:
                p.grad = None

class SGD(Optimizer):
    def __init__(self, parameters, lr=0.01, momentum=0.0):
        super().__init__(parameters, lr)
        self.momentum = momentum
        self.velocities = {id(p): 0 for p in self.parameters}

    def step(self):
        i=1
        for p in self.parameters:
            if p.grad is None:
                continue
            if i % 2 == 0:
                p.grad = p.grad.sum(axis=0)
            # Ensure that velocity matches the shape of the parameter
            p.grad = Tensor(p.grad)
            # velocity = - (p.grad * self.lr) + (self.velocities[id(p)] * self.momentum)
            # self.velocities[id(p)] = velocity
            # Update the parameter with the velocity
            # p.data = velocity.data + p.data
            p.data -= p.grad.data * self.lr
            i+=1