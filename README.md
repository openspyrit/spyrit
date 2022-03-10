![GitHub tag (latest by date)](https://img.shields.io/github/v/tag/openspyrit/spyrit?logo=github)
[![GitHub](https://img.shields.io/github/license/openspyrit/spyrit?style=plastic)](https://github.com/openspyrit/spyrit/blob/master/LICENSE.md)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/spyrit.svg)](https://pypi.python.org/pypi/spyrit/)
[![Docs](https://readthedocs.org/projects/spyrit/badge/?version=latest&style=flat)](https://spyrit.readthedocs.io/en/master/)

# Spyrit
SPYRIT is a [PyTorch](https://pytorch.org/)-based toolbox for deep image reconstruction. While SPYRIT was originally designed for single-pixel image reconstruction, it can solve any linear reconstruction problem.
   
## Getting Started

### User mode

The spyrit package is available for Linux, MacOs and Windows. You can install it with pypi (we recommend you to use virtual environment).

#### Linux and MacOs

```
pip install spyrit
```

#### Windows

On Windows you need first to install [torch](https://pytorch.org/get-started/locally/). Adapt to your configuration. Two examples below.

CPU version using `pip `

```
pip install requests torch==1.8.0+cpu torchvision==0.9.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
```

GPU version using `conda` 

``` shell
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
```

Then install SPyRiT using `pip`

```shell
pip install spyrit
```

### Developer mode

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

First, you need to clone the repository:

```
git clone https://github.com/openspyrit/spyrit.git
```

Then, you can install the spyrit package with python (we recommend you to use virtual environment)

#### Linux and MacOs

```
cd spyrit
pip install -e .
```

#### Windows

On Windows you need first to install [torch](https://pytorch.org/get-started/locally/). Here it's cpu version, adapt to your configuration. 

NB: It may be necessary to run the following commands using administrator rights (e.g., starting your Python environment with administrator rights).

```
cd spyrit
pip install requests torch==1.8.0+cpu torchvision==0.9.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
pip install -e .
```

#### Versioning

To change the version of the package on [pypi](https://pypi.org/project/spyrit/), you need to:
 - change the version in [setup.py](https://github.com/openspyrit/spyrit/blob/master/setup.py#L45) to new_version
 - ```git commit setup.py -m "Towards new_version"```
 - ```git tag -a new_version -m "new_version"```
 - ```git push --follow-tags```

## API Documentation
https://spyrit.readthedocs.io/

## Prerequisites

All the necessary packages and libraries are contained within the ```setup.py ``` file.

- numpy
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

A minimal exemple can be found [here](https://github.com/openspyrit/spyrit/blob/master/.github/workflows/example.py). To run it, clone or download the file and you can do:

```
python example.py
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

* [Jin LI](https://github.com/happyjin/ConvGRU-pytorch) for his implementation of Convolutional Gated Recurrent Units for PyTorch
* [Erik Lindernoren](https://github.com/eriklindernoren/Action-Recognition) for his processing of the UCF-101 Dataset.
