
## Class and Object Diagrams

### Version 0:

![First Version of the Class Diagram](assets/v0.jpg)

This initial version of the class diagram was extended from the UML we have seen in class and constructed according to the description given for each entity. It was later refined following the user interface that we must use and the implementation of the code. 

## Implementation Explanation:

### User Interface:

```python
 class Model(Module):
    def __init__(self)-> None:
        self.linear1 = Linear(32*32, 20)
        self.linear2 = Linear(20, 10)

    def forward(self, x):
        x = self.linear1(x)
        return self.linear2(x)

 model = Model()
 print(model)

 optimiser = SGD(model.parameters() , lr=0.01, momentum=0.9)
 loss_fn = MSE()
 dataset = MNIST()
 dataloader = DataLoader(dataset, batchsize=32, transform=None)

 for x, y in dataloader:
    optimizer.zero_grad()
    y_hat = model(x)
    loss = loss_fn(y, y_hat)
    loss.backward()
    optimizer.step()
```

To have this user interface, we implemented:

### `class Module`, `Linear`, `Parameter`, `Activation`:
Since the parameters corresponding to a layer should be registered as soon as the layer is declared, and the optimizer should be able to access them through `model.parameters()`, this was handled by the `Module` class because the `Model` class is a subclass of `Module`. Also, the `Linear` class should inherit from `Module` to save its parameters in the `Module` class.

So the `Module` class has as attributes two dictionaries, one to save the subclasses and another to save the parameters.

The `Linear` class takes as inputs the `input_size` and `output_size`. In the constructor, the weights and bias are initialized as `Parameter` type (Which is a `Tensor` with `requires_grad=True` and `is_leaf=True`). Weights are randomly initialized to random values - 0.5 to be centered around 0, and the bias is initialized to 0.

The `__setattr__` method was modified inside the `Module` class as follows:
```python
def __setattr__(self, name, value): 
    if isinstance(value, Module):
        self._subclasses[name]=value
    if isinstance(value, Parameter): 
        self._parameters[name]=value
    super().__setattr__(name, value)
```
This way, when a linear layer is declared, ex: `Linear(32*32, 20)`, setting the attributes weights and bias which are of type `Parameter` will register them in the dictionary `_parameters` of the `Module` class.

When the model is declared, the different linear layers set as attributes, ex: `self.linear1 = Linear(32*32, 20)` are registered in the `_subclasses` dictionary of the `Module` class.

`super().__setattr__(name, value)` is used to normally set the attributes in the class dictionary, so that when we call `model.linear1` for example, it returns the linear layer using the default `__getattribute__` method.

Also, the `Activation` class was implemented to inherit from `Module` to save it in the `_subclasses` dictionary of the `Module` class, so that when we print the model to visualize its architecture, we can see the different layers and activation functions used in the model, and so the `__repr__` method was implemented in the `Module` class to print the model architecture:
```python
def __repr__(self):
    return f"{self.__class__.__name__}({_subclasses})"
```
with `__repr__` implemented also in the subclasses to have a representation of the layers and activation functions used in the model.

In the `Module` class, the method `__call__` was implemented to call the `forward` method presented as an abstract method in `Module`, and implemented by the subclasses (`Model`, `Linear`, `Activation`). In this way when we use: `model(x)`, the `forward` method of the model is called, which in its turn calls the `forward` method of the linear layers (when `self.linear1(x)` is called for example) and activation functions used in the model.

The `forward` method of `Linear` applies the affine linear transformation to the input `x` and outputs `y = xW^T + b`, where `W` is the weights and `b` is the bias. The `forward` method of `Activation` is an abstract method that is implemented by the different activation functions (`ReLU`, `Softmax`) that inherit `Activation` class.