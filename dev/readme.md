# library development

## Set-up utils

- [x] **[setup.py](../setup.py)**: for packaging the library, saves version, dependencies and other metadata  
- [x] **[requirements.txt](../requirements.txt)**: for installing dependencies  
- [x] **[dev/](.)**: for development scripts and utilities including versioning and directory structure 
    - [x] `set-pythonpath.sh`: for setting the python path to the library directory during developming to allow imports correctly, make sure you wun it though `source` to allow the changes to take effect in the current shell (saving $PYTHONPATH as a global variable)  
    ```bash
    source ./dev/set-pythonpath.sh 
    ```
    - [x] `update-version.sh`: for updating the version of the library, it prompts version input from user (x.y.z format where x is major, y is minor and z is patch) and updates the version number in [VERSION file](../VERSION) + description of updates in [CHANGELOG.md](./CHANGELOG.md)

Under development, download the library from teh project's directory by running
```bash
pip install -e .
```
This will look for the `setup.py` file, downlaod any dependencies and install the library in the current environment in editable mode.
```bash
Defaulting to user installation because normal site-packages is not writeable
Obtaining file:///mnt/g/my_stuff/masters/saclay/courses/M1/Object-Orietnted-Programming/project
  Preparing metadata (setup.py) ... done
Collecting matplotlib==3.4.3
  Downloading matplotlib-3.4.3.tar.gz (37.9 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 37.9/37.9 MB 1.7 MB/s eta 0:00:00
  Preparing metadata (setup.py) ... done
Collecting numpy==2.0.2
  Downloading numpy-2.0.2-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (19.5 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 19.5/19.5 MB 4.0 MB/s eta 0:00:00
Collecting opencv-contrib-python==4.10.0.84
  Downloading opencv_contrib_python-4.10.0.84-cp37-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (68.7 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 68.7/68.7 MB 2.8 MB/s eta 0:00:00
Collecting pandas==2.2.3
  Downloading pandas-2.2.3-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (13.1 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 13.1/13.1 MB 4.7 MB/s eta 0:00:00
Collecting scikit-learn==1.5.2
  Downloading scikit_learn-1.5.2-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (13.3 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 13.3/13.3 MB 6.0 MB/s eta 0:00:00
Collecting cycler>=0.10
  Downloading cycler-0.12.1-py3-none-any.whl (8.3 kB)
Collecting kiwisolver>=1.0.1
  Downloading kiwisolver-1.4.8-cp310-cp310-manylinux_2_12_x86_64.manylinux2010_x86_64.whl (1.6 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.6/1.6 MB 3.6 MB/s eta 0:00:00
Collecting pillow>=6.2.0
  Downloading pillow-11.1.0-cp310-cp310-manylinux_2_28_x86_64.whl (4.5 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 4.5/4.5 MB 5.1 MB/s eta 0:00:00
Requirement already satisfied: pyparsing>=2.2.1 in /usr/lib/python3/dist-packages (from matplotlib==3.4.3->ann==1.0.0) (2.4.7)
Collecting python-dateutil>=2.7
  Downloading python_dateutil-2.9.0.post0-py2.py3-none-any.whl (229 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 229.9/229.9 KB 10.3 MB/s eta 0:00:00
Collecting tzdata>=2022.7
  Downloading tzdata-2025.1-py2.py3-none-any.whl (346 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 346.8/346.8 KB 5.9 MB/s eta 0:00:00
Collecting pytz>=2020.1
  Downloading pytz-2025.1-py2.py3-none-any.whl (507 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 507.9/507.9 KB 6.4 MB/s eta 0:00:00
Collecting threadpoolctl>=3.1.0
  Downloading threadpoolctl-3.5.0-py3-none-any.whl (18 kB)
Collecting scipy>=1.6.0
  Downloading scipy-1.15.1-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (40.6 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 40.6/40.6 MB 3.3 MB/s eta 0:00:00
Collecting joblib>=1.2.0
  Downloading joblib-1.4.2-py3-none-any.whl (301 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 301.8/301.8 KB 3.4 MB/s eta 0:00:00
Requirement already satisfied: six>=1.5 in /usr/lib/python3/dist-packages (from python-dateutil>=2.7->matplotlib==3.4.3->ann==1.0.0) (1.16.0)
Building wheels for collected packages: matplotlib
  Building wheel for matplotlib (setup.py) ... done
  Created wheel for matplotlib: filename=matplotlib-3.4.3-cp310-cp310-linux_x86_64.whl size=10425356 sha256=526733cbdbd106ae89b8ad8f91cffcc916d30b1a57cb978d9c0f8b1919e4547a
  Stored in directory: /home/raysas/.cache/pip/wheels/71/af/e4/d399b616d3e7ae88374c2ebab2d5d3ecf776a3590d4f5f768f
Successfully built matplotlib
Installing collected packages: pytz, tzdata, threadpoolctl, python-dateutil, pillow, numpy, kiwisolver, joblib, cycler, scipy, pandas, opencv-contrib-python, matplotlib, scikit-learn, ann
  WARNING: The scripts f2py and numpy-config are installed in '/home/raysas/.local/bin' which is not on PATH.
  Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
  Running setup.py develop for ann
Successfully installed ann cycler-0.12.1 joblib-1.4.2 kiwisolver-1.4.8 matplotlib-3.4.3 numpy-2.0.2 opencv-contrib-python-4.10.0.84 pandas-2.2.3 pillow-11.1.0 python-dateutil-2.9.0.post0 pytz-2025.1 scikit-learn-1.5.2 scipy-1.15.1 threadpoolctl-3.5.0 tzdata-2025.1
```

> [!NOTE]
changed the library name from `ann` to `che3le`, this action is reversible anytime by running `./dev/change-libname.sh` script. However, in order for the example test to wrk need to manually change the imports in the `example/MNIST_model.py` file from `che3le` to whichever name is set for the library. 

```bash
./dev/change-libname.sh 
```

```text
>>> Enter the library relative path: ann
>>> Enter the new library name: che3le

>>> Changing library name from 'ann' to 'che3le'

  >>> '/mnt/g/my_stuff/masters/saclay/courses/M1/Object-Orietnted-Programming/project/ann/extensions/dataloader.py' is updated
  >>> '/mnt/g/my_stuff/masters/saclay/courses/M1/Object-Orietnted-Programming/project/ann/extensions/dataset.py' is updated
  >>> '/mnt/g/my_stuff/masters/saclay/courses/M1/Object-Orietnted-Programming/project/ann/extensions/transforms.py' is updated
  >>> '/mnt/g/my_stuff/masters/saclay/courses/M1/Object-Orietnted-Programming/project/ann/extensions/__init__.py' is updated
  >>> '/mnt/g/my_stuff/masters/saclay/courses/M1/Object-Orietnted-Programming/project/ann/nn/activation.py' is updated
  >>> '/mnt/g/my_stuff/masters/saclay/courses/M1/Object-Orietnted-Programming/project/ann/nn/linear.py' is updated
  >>> '/mnt/g/my_stuff/masters/saclay/courses/M1/Object-Orietnted-Programming/project/ann/nn/loss.py' is updated
  >>> '/mnt/g/my_stuff/masters/saclay/courses/M1/Object-Orietnted-Programming/project/ann/nn/module.py' is updated
  >>> '/mnt/g/my_stuff/masters/saclay/courses/M1/Object-Orietnted-Programming/project/ann/nn/optimizer.py' is updated
  >>> '/mnt/g/my_stuff/masters/saclay/courses/M1/Object-Orietnted-Programming/project/ann/nn/parameter.py' is updated
  >>> '/mnt/g/my_stuff/masters/saclay/courses/M1/Object-Orietnted-Programming/project/ann/nn/__init__.py' is updated
  >>> '/mnt/g/my_stuff/masters/saclay/courses/M1/Object-Orietnted-Programming/project/ann/tensor.py' is updated
  >>> '/mnt/g/my_stuff/masters/saclay/courses/M1/Object-Orietnted-Programming/project/ann/utils/evaluation.py' is updated
  >>> '/mnt/g/my_stuff/masters/saclay/courses/M1/Object-Orietnted-Programming/project/ann/utils/functions.py' is updated
  >>> '/mnt/g/my_stuff/masters/saclay/courses/M1/Object-Orietnted-Programming/project/ann/utils/processing.py' is updated
  >>> '/mnt/g/my_stuff/masters/saclay/courses/M1/Object-Orietnted-Programming/project/ann/utils/simulation.py' is updated
  >>> '/mnt/g/my_stuff/masters/saclay/courses/M1/Object-Orietnted-Programming/project/ann/utils/validation.py' is updated
  >>> '/mnt/g/my_stuff/masters/saclay/courses/M1/Object-Orietnted-Programming/project/ann/utils/__init__.py' is updated
  >>> '/mnt/g/my_stuff/masters/saclay/courses/M1/Object-Orietnted-Programming/project/ann/__init__.py' is updated
  >>> '/mnt/g/my_stuff/masters/saclay/courses/M1/Object-Orietnted-Programming/project/setup.py' is updated

 >>> Library name is updated from 'ann' to 'che3le' <<<
```

(che3learn is an interesting name)


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