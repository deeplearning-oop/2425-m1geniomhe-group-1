from abc import ABC, abstractmethod

class Optimizer(ABC):
    def __init__(self, parameters, lr=0.01, momentum=0):
        self.parameters = [p for p in parameters]
        self.lr = lr
        self.momentum = momentum
        
    @abstractmethod
    def step(self):
        raise NotImplementedError

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
        for p in self.parameters:
            # p.data -= p.grad * self.lr
            adjustment = - (p.grad * self.lr)
            velocity = adjustment + (self.velocities[id(p)] * self.momentum)
            self.velocities[id(p)] = velocity
            p.data += velocity