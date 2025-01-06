'''
setting up our library
'''

from setuptools import setup, find_packages

with open("VERSION") as version_file:
    version = version_file.read().strip()

with open("docs/readme.md", "r", encoding="utf-8") as readme_file:
    long_description = readme_file.read()

setup(
    name='ann',
    version=version,
    description='Learning OOP the Hard Way: Modeling and Implementing a Deep Learning Library',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Yazid Hoblos, Joelle Assys, Rayane Adam',
    url='https://github.com/deeplearning-oop/2425-m1geniomhe-group-1/',
    license='MIT',
    packages=find_packages(),
    python_requires='>=3.6',
    install_requires=[
        line.strip() for line in open("requirements.txt").readlines() if line.strip()
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)
