![GitHub tag (latest by date)](https://img.shields.io/github/v/tag/openspyrit/spyrit?logo=github)
[![GitHub](https://img.shields.io/github/license/openspyrit/spyrit?style=plastic)](https://github.com/openspyrit/spyrit/blob/master/LICENSE.md)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/spyrit.svg)](https://pypi.python.org/pypi/spyrit/)
[![Docs](https://readthedocs.org/projects/spyrit/badge/?version=latest&style=flat)](https://spyrit.readthedocs.io/en/master/)

# SPyRiT
SPyRiT is a [PyTorch](https://pytorch.org/)-based toolbox for deep image reconstruction. While SPyRiT was originally designed for single-pixel image reconstruction, it can solve any linear reconstruction problem.
   
# Installation
The spyrit package is available for Linux, MacOs and Windows. We recommend to use a virtual environment.
## Linux and MacOs
(user mode)
```
pip install spyrit
```
(developper mode)
```
git clone https://github.com/openspyrit/spyrit.git
cd spyrit
pip install -e .
```

## Windows
On Windows you may need to install PyTorch first. It may also be necessary to run the following commands using administrator rights (e.g., starting your Python environment with administrator rights).

Adapt the two examples below to your configuration (see [here](https://pytorch.org/get-started/locally/) for the latest instructions)

(CPU version using `pip`)

```
pip3 install torch torchvision torchaudio
```

(GPU version using `conda`)

``` shell
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

Then, install SPyRiT using `pip`.

## Test
To check the installation, run in your python terminal:
```
import spyrit
```

## Examples
Simple reconstruction examples can be found in [spyrit/test/tuto_core_2d.py](https://github.com/openspyrit/spyrit/blob/master/spyrit/test/tuto_core_2d.py).

# API Documentation
https://spyrit.readthedocs.io/

# Authors (alphabetical order)
* Thomas Baudier
* Sebastien Crombez
* Nicolas Ducros - [Website](https://www.creatis.insa-lyon.fr/~ducros/WebPage/index.html)
* Antonio Tomas Lorente Mur - [Website](https://www.creatis.insa-lyon.fr/~lorente/)
* Fadoua Taia-Alaoui

# License
This project is licensed under the Creative Commons Attribution Share Alike 4.0 - see the [LICENSE.md](LICENSE.md) file for details

# Acknowledgments
* [Jin LI](https://github.com/happyjin/ConvGRU-pytorch) for his implementation of Convolutional Gated Recurrent Units for PyTorch
* [Erik Lindernoren](https://github.com/eriklindernoren/Action-Recognition) for his processing of the UCF-101 Dataset.
