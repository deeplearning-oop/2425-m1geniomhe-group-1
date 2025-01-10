# library development

## Set-up utils

- [x] **[setup.py](../setup.py)**: for packaging the library, saves version, dependencies and other metadata  
- [x] **[requirements.txt](../requirements.txt)**: for installing dependencies  
- [x] **[dev/](.)**: for development scripts and utilities including versioning and directory structure 
    - [x] `set-pythonpath.sh`: for setting the python path to the library directory during developming to allow imports correctly, make sure you wun it though wither `source` or `.` to allow the changes to take effect in the current shell (saving $PYTHONPATH as a global variable)  
    ```bash
    ./dev/set-pythonpath.sh ann/
    ```
    - [x] `update-version.sh`: for updating the version of the library, it prompts version input from user (x.y.z format where x is major, y is minor and z is patch) and updates the version number in [VERSION file](../VERSION) + description of updates in [CHANGELOG.md](./CHANGELOG.md)

## Resources

### libraries writing  
_+ documenting on github_    

- [private methods in python](https://www.datacamp.com/tutorial/python-private-methods-explained)  
- [python packaging](https://packaging.python.org/en/latest/tutorials/packaging-projects/)  
- [python app in github (by github)](https://docs.github.com/en/actions/use-cases-and-examples/building-and-testing/building-and-testing-python)   
- [python package on github tutorial](https://qbee.io/docs/tutorial-github-python.html)   
- [importing modules in python](https://www.datacamp.com/tutorial/modules-in-python?dc_referrer=https%3A%2F%2Fwww.google.com%2F)  
- [modules in python documentation](https://docs.python.org/3/tutorial/modules.html)  
- [stack overflow: import module issue](https://stackoverflow.com/questions/9383014/cant-import-my-own-modules-in-python)   
- [github docum: diagrams in markdown](https://docs.github.com/en/get-started/writing-on-github/working-with-advanced-formatting/creating-diagrams)  

### pytorch

* [pytorch documentation](https://pytorch.org/docs/stable/index.html)  
* [pytorch github repo](https://github.com/pytorch/pytorch)
* [pytorch for DL](https://www.learnpytorch.io/)  

### datasets

> [!TIP]
> _To load large datasets, need to actually download them in a directory and then load them in the notebook by accessing a deafult path name which we have assigned in the implementation. e.g., download MNIST from web in data/MNIST can save images and labels each in a subdirectory, when we load we actually go through the files and convert them to tensors_

* [CO large datasets download](https://oyyarko.medium.com/google-colab-work-with-large-datasets-even-without-downloading-it-ae03a4d0433e)   
* [MNIST official database to accesss](https://yann.lecun.com/exdb/mnist/)



### tools

- [figma](https://www.figma.com/) for designing images in explanation of the library design  
- [diagrams.net](https://app.diagrams.net/) for designing the class diagrams of the library  
- [carbon](https://carbon.now.sh/) for designing the code snippets in the documentation  