# Section 1:

## Class and Object Diagrams

### Version 0:

![First Version of the Class Diagram](diagrams/v0-light.jpg)

This initial version of the class diagram was constructed according to the description given for each entity, similar to the UML we have seen in class. It was later refined following the user interface that we must use and the implementation of the code. 

### Final Version:

#### Class Diagram:

![Final Version of the Class Diagram](diagrams/class-v1.jpg)

- class `Module` defined as abstract class will serve as a base class for `Model`, `Linear`, `Activation`, and - `Loss` to enforce them to implement the abstract forward method, so these classes have inheritance relationships with `Module`.  
- class `Model` has a composition relationship with `Linear' since we assumed that if we deleted a model its linear layers should not exist on their own. It is 1 to 1..M relationship: 1 model should have at least 1 linear layer, and a defined linear layer should belong to only one model.
- class `Model` has a composition relationship with `Activation`. It is 1 to many relationship. A model can have 0 or many activation functions, and a defined activation function can belong to one model.
- class 'Linear' has 1 to many relationship with `Parameter`. There is two associations: one for the weights and one for the bias. For the weights it is 1 to 1, and for the bias it is 1 to 0..1 since the bias is optional. 
- class `Parameter` inherits `Tensor`. 
- class `Optimizer` updates the parameters of the model. So it has 1 to many relationship with `Parameter`. 
- class `Activation` is defined as an abstract class to enforce the activation functions `Sigmoid`, `ReLU`, and `Softmax`, childs of `Activation`, to implement the forward method.
- class `Loss` is defined as an abstract class to enforce the loss functions `MSE`, `CrossEntropy`, and `NegativeLogLikelihood`, childs of `Loss`, to implement the forward method. 
- class `Dataset` is defined as an abstract class to enforce the childs like 'MNIST' to implement the abstract method `__getitem__` and `__len__`. It has two associations with `Tensor`: one for the data and one for the target, 1 to 1 relationship. It has one to many relationship with `DataLoader` since a dataset can be loaded by 0 or many dataloaders, and a dataloader can load 1 dataset. It has also two associations with `Transform`: one for transformation of the data and one for the target, 1 to 0..1 relationship assuming that we create a new transform each time we want to transform the data or the target of a given dataset and we can have a dataset without any transformation or 1 transformation that can be `Compose` if we want to apply multiple transformations.
- class `ToTensor`, `MinMax`, `Normalize` are the transformations that can be applied to the data and target. They inherit from `Transform`. `Standardize` is a child of `Normalize`.
- class `Compose` is a transformation that can apply multiple transformations to the data and target. It has a 1 to many relationship with `Transform` since it can apply 0 or many transformations. It inherits from Transform and it has an aggregation relationship with `Transform` since it contains a list of transformations. 

#### Object Diagram:

![Final Version of the Object Diagram](diagrams/object-v1.jpg)

To provide an object diagram, we took a specific example. Note that the abstract classes we defined cannot be instantiated as objects. So the object diagram contains only the entities that can be instantiated as objects with their attributes. 
We considered a model with mode=Training and a dataset of type MNIST with train=True that have data and target as Tensor type. A dataloader with batch_size=32 and shuffle=True is used to load the dataset, and transform3 of type Compose is applied to the dataset and this transform3 is composed of two other transforms, transform1 of type ToTensor and transform2 of type Standardize. The model has two linear layers, one ReLU activation function to be applied after the first linear layer and a second activation function: Softmax to be applied to the output layer. linear1 has W1 and b1 as parameters, and linear 2 has W2 and b2 as parameters. All the parameters will be accessed by the optimiser of type SGD with learning_rate=0.01 and momentum=0.9 to update them. A loss_fn of type MSE is used. Only direct relationships are shown in the object diagram similar to the class diagram, but the model will have access to the data by using the dataloader and the loss function will be applied to the output of the model and the target of the dataset. 

## General Approach for library modeling and implementation:

Yazid 

# Section 2:

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



# Section 3:


# Section 4:
