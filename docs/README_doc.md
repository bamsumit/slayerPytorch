# README

This readme is here to help updating the documentation for SLAYER PyTorch.

## Install the sphinx tool

Start with sourcing the same virtual environment where you installed SLAYER. Then, install **sphinx** (more documentation can be found [here](https://www.sphinx-doc.org/en/master/index.html)) by running:
```
$ pip install sphinx
```
or with the option -U to install in the user space.

**NB**: it is required to install sphinx in the same virtual environment as slayer to enable it to find the code and do the links in the documentation.

## Update the documentation

If you made any change, you can update the documentation with the following
``` 
$ cd <path_to_slayerPytorch>/docs
$ make html
```
To build pdf documentation,
```
$ make latexpdf
```
Sphinx pdf build uses `latexmk`. If it is not installed in your system, install using
```
$ sudo apt-get update -y
$ sudo apt-get install -y latexmk
```

