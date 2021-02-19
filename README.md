# Spyrit Version 0.1

Spyrit Toolbox aims to provide all the necessary tools for single-pixel imaging. Starting from simulation, reconstruction, and interface with DMD and spectrometers.
The aim of this toolbox is to cover all aspects of single-pixel imaging : from simulation to experimental, we aim to provide tools to make realistic measurements and provide reconstruction algorithms. 
    
## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Under Linux
```
git clone --recurse-submodules https://github.com/openspyrit/spyrit.git
```

### Prerequisites

All the necessary packages and libraries are contained within the ```setup.py ``` file.

- numpy (>1.3.0)',
- matplotlib (>2.2.4)',
- scipy (>1.1.0)',
- torch (>1.1.0)',
- torchvision (>0.2.2)',
- PIL (>5.3.0)',
- cv2 (>4.0.0)',
- imutils (>0.5.3)',
- pywt (>1.0.1)',
- fht=['https://github.com/nbarbey/fht'] (included as a submodule in spyrit/fht),


### Installing

A step by step series of examples that tell you how to get a development env running

Say what the step will be

```
python3 setup.py
```

To check that the installation has been a success, try running the following lines in yout python terminal :


```
import spyrit
```

End with an example of getting some data out of the system or using it for a little demo

```
import torch;
a = torch.randn(64,64);
```


## Running the tests

Explain how to run the automated tests for this system

### Break down into end to end tests

Explain what these tests test and why

```
Give an example
```

### And coding style tests

Explain what these tests test and why

```
Give an example
```

## Deployment

Add additional notes about how to deploy this on a live system

## Built With

* [Dropwizard](http://www.dropwizard.io/1.0.2/docs/) - The web framework used
* [Maven](https://maven.apache.org/) - Dependency Management
* [ROME](https://rometools.github.io/rome/) - Used to generate RSS Feeds

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


