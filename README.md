# OOP the hard way: Deep Learning Library
_This is the project for the Object Oriented Programming course at the University of Paris-Saclay, M1 GENIOMHE: 2425-m1geniomhe-group-1_  
_Check out this quick demo:_   <a href="https://colab.research.google.com/github/deeplearning-oop/2425-m1geniomhe-group-1/blob/main/example/MNIST_model.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open Project in colab"/></a> 

[![GitHub](https://img.shields.io/badge/GitHub-deeplearning_oop_org-black)](https://github.com/deeplearning-oop/2425-m1geniomhe-group-1)[![Version](https://img.shields.io/badge/version-1.0.1-grassgreen)](./dev/changelog.md)  ![Build](https://img.shields.io/badge/installation-pip-brightgreen) [![Python](https://img.shields.io/badge/python-3.+-blue)](https://www.python.org/downloads/release/python-390/)  [![License](https://img.shields.io/badge/license-MIT-orange)](./LICENSE.md)  [![Submission](https://img.shields.io/badge/submission-successful_:\)-yellow)](./docs/readme.md)

## Description
This project is a deep learning library that allows the user to create and train artificial neural networks. The library is implemented in Python and uses the NumPy library for matrix operations. The aim of this project is to perform the task using object-oriented programming principles and design patterns. The user-interface and performance were consequently compared to the PyTorch[^1] library. The library relies on gradient calculation and backpropagation through a computational graph. The library supports the following features:
- Fully connected layers
- Activation functions (ReLU, Sigmoid, Softmax)
- Loss functions (MSE, Cross-Entropy, Negative Log-Likelihood) 
- Optimizers (SGD)  

> [!NOTE]
> The implementation of library is based on OOP principles such as encapsulation, inheritance, and abstraction with the use of properties, magic dunder methods and decorators.

For a more detailed description refer to the [docs readme](./docs/readme.md).

[^1]: https://pytorch.org/

## Installation

To install and test the library, install using pip:
```bash
pip install git+https://github.com/deeplearning-oop/2425-m1geniomhe-group-1.git
```
It will install all dependencies found in [`requirements.txt`](requirements.txt), and the library will be available for use through a simple import from the Python environment where it was installed:

```python
import che3le
help(che3le) # this will display a short description of the modules and classes available
```


## Project proposal
Take a look at the project description proposed to the students in [2425-project-proposal.pdf](2425-project-proposal.pdf).

### Report
You can find the description of the deep learning library in the [docs](./docs) folder.

### Submission

Submitted a report and a presentation.
The code submission is in the src folder. 
The overall directory structure is as follows (dating 08/01/2025):

```text
2425-m1geniomhe-group-1/
├── 2425-project-proposal.pdf
├── LICENSE.md  # -- MIT License
├── README.md
├── VERSION # -- version file to keep tracks of updates
├── ann/   # -- the python library that is set up by setup.py (pip install)
│   ├── __init__.py
│   ├── extensions/
│   │   ├── __init__.py
│   │   ├── dataloader.py
│   │   ├── dataset.py
│   │   └── transforms.py
│   ├── nn/
│   │   ├── __init__.py
│   │   ├── activation.py
│   │   ├── linear.py
│   │   ├── loss.py
│   │   ├── module.py
│   │   ├── optimizer.py
│   │   └── parameter.py
│   ├── tensor.py
│   └── utils/
│       ├── __init__.py
│       ├── functions.py
│       ├── processing.py
│       ├── simulation.py
│       └── validation.py
├── assets/ # -- images 
│   ├── comparison1.png
│   ├── comparison2.png
│   ├── comparison3.png
│   ├── comparison4.png
│   ├── repo_structure.png
│   └── data_viz.png
├── docs/
│   ├── diagrams/
│   │   ├── class-v1.jpg
│   │   ├── object-v1.jpg
│   │   ├── v0-dark.jpg
│   │   └── v0-light.jpg
│   ├── readme.md
│   └── report.pdf
├── example/
│   └── MNIST_model.py
├── requirements.txt
├── setup.py
├── src/
│   ├── activation.py
│   ├── dataloader.py
│   ├── dataset.py
|   ├── linear.py
│   ├── loss.py
│   ├── module.py
│   ├── optimizer.py
│   ├── parameter.py
│   ├── tensor.py
│   └── transforms.py
└── tests/
|   ├── benchmarks/
|   |   ├── gene_expression.py
|   │   └── MNIST.py
|   ├── unit/  # -- unti tests in classes and comparison with pytorch 
|   |   ├── vision_transforms.ipynb
|   │   └── vision.ipynb
|   ├── mocks/  # -- checking consistency with pytorch 
|   |   ├── simulated_linear.ipynb
|   |   ├── simulated_circular.ipynb
|   |   ├── simulated_checkerboard.ipynb
|   │   └── simulated_spiral.ipynb
|   └── backup/ 
|       ├── test_simulation.py
|       └── test_with_dataloaders.py
└── utils
    ├── changelog.md
    ├── make-global.sh
    ├── readme.tree
    └── update-version.sh
```

## Demo

Kindly find demos in the [example](./example) folder.
If you install the library locally, you can run the following script to train a simple model on the MNIST dataset:
```bash
python example/MNIST_model.py
```

You can test it as well on [Google Colab by clicking on this link]() and running the notebook.

## Authors
* Joelle ASSY ([@joelleas](https://github.com/joelleas))
* Yazid HOBLOS ([@yazid-hoblos](https://github.com/yazid-hoblos))  
* Rayane ADAM ([@raysas](https://github.com/raysas))
