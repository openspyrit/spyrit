
![GitHub tag (latest by date)](https://img.shields.io/github/v/tag/openspyrit/spyrit?logo=github)
[![GitHub](https://img.shields.io/github/license/openspyrit/spyrit?style=plastic)](https://github.com/openspyrit/spyrit/blob/master/LICENSE.md)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/spyrit.svg)](https://pypi.python.org/pypi/spyrit/)

# Spyrit

Spyrit Toolbox aims to provide all the necessary tools for single-pixel imaging. Starting from simulation, reconstruction, and interface with DMD and spectrometers.
The aim of this toolbox is to cover all aspects of single-pixel imaging : from simulation to experimental, we aim to provide tools to make realistic measurements and provide reconstruction algorithms. 
    
## Getting Started

### User mode

The spyrit package is available for Linux, MacOs and Windows. You can install it with pypi (we recommend you to use virtual environment).

#### Linux and MacOs

```
pip install spyrit
```

#### Windows

On Windows you need first to install [torch](https://pytorch.org/get-started/locally/). Here it's cpu version, adapt to your configuration.

```
pip install requests torch==1.8.0+cpu torchvision==0.9.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
pip install spyrit
```

### Developper mode

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

First, you need to clone the repository

```
git clone --recurse-submodules https://github.com/openspyrit/spyrit.git
```

Then, you can install the spyrit package with python (we recommend you to use virtual environment)

#### Linux and MacOs

```
cd spyrit
pip install -e .
```

#### Windows

On Windows you need first to redo the symbolic link to fht inside the spyrit repository and then to install [torch](https://pytorch.org/get-started/locally/). Here it's cpu version, adapt to your configuration.

```
cd spyrit
rm -r -fo fht
cmd /c mklink /d fht spyrit\fht\fht
pip install requests torch==1.8.0+cpu torchvision==0.9.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
pip install -e .
```

#### Versioning

To change the version of the package on [pypi](https://pypi.org/project/spyrit/), you need to:
 - change the version in [setup.py](https://github.com/openspyrit/spyrit/blob/master/setup.py#L45) to new_version
 - ```git commit setup.py -m "Towards new_version"```
 - ```git tag -a new_version -m "new_version"```
 - ```git push --follow-tags```

## Prerequisites

All the necessary packages and libraries are contained within the ```setup.py ``` file.

- numpy (==1.19.3)
- matplotlib
- scipy
- torch
- torchvision
- Pillow
- opencv-python
- imutils
- PyWavelets
- wget
- imageio
- [fht](https://github.com/nbarbey/fht) (included as a submodule in spyrit/fht),

## Test

To check that the installation has been a success, try running the following lines in yout python terminal :


```
import spyrit
```

End with an example of getting some data out of the system or using it for a little demo

```
import torch;
a = torch.randn(64,64);
```

A minimal exemple can be found [here](https://github.com/openspyrit/spyrit/blob/master/.github/workflows/example.py). To run it, you can do

```
cd spyrit
python .github/workflows/example.py
```

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct, and the process for submitting pull requests to us.


## Authors

* **Antonio Tomas Lorente Mur** - *Initial work* - [Website](https://www.creatis.insa-lyon.fr/~lorente/)
* **Nicolas Ducros** - *Initial work* - [Website](https://www.creatis.insa-lyon.fr/~ducros/WebPage/index.html)
* **Sebastien Crombez** - *Initial work* - [Website]


## License

This project is licensed under the Creative Commons Attribution Share Alike 4.0 - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* [Nicolas Barbey](https://github.com/nbarbey/fht) for his Fast Hadamard Transform implementation in python  
* [Jin LI](https://github.com/happyjin/ConvGRU-pytorch) for his implementation of Convolutional Gated Recurrent Units for PyTorch
* [Erik Lindernoren](https://github.com/eriklindernoren/Action-Recognition) for his processing of the UCF-101 Dataset.


