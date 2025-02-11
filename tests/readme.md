# Library tests

This directory contains code tests of this library and pytorch models.  
It's split into different directories for different types of tests:

| Directory | Description |
| --- | --- |
| `benchmark` |  models tested on benchmark data like MNIST |  
| `unit` |  code to perform unit tests on some classes while portraying pytorch semantics in parallel to compare interface and behavior |  
| `backup` | old tests done while developing the library for reference |  

> [!IMPORTANT]
_The tests in these directories is performed before compellty packaging the library, it uses the code from `src/` and does not import the library directly from code. They're left public for reference and to show the testing process._ For that purpose we created an `example/` directory in the main project directory to show how to use the library after installation.