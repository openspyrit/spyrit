from setuptools import setup, find_packages, Extension
from os.path import join
import os
import sys
from sys import platform as _platform

def readme():
    with open("README.md") as f:
        return f.read()

setup(name='spyrit',
      version='2.1.0',
      description='Demo package',
      url='https://github.com/openspyrit/spyrit',
      long_description = readme(),
      long_description_content_type = "text/markdown",
      classifiers = [
        "Development Status :: 5 - Production/Stable",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent"
      ],
      author='Antonio Tomas Lorente Mur, Nicolas Ducros, Sebastien Crombez',
      author_email='Nicolas.Ducros@insa-lyon.fr',
      keywords = "tutorial package",
      license='Attribution-ShareAlike 4.0 International',
      python_requires='>=3.6',
      install_requires=[
          'numpy',
          'matplotlib',
          'scipy',
          'torch',
          'torchvision',
          'Pillow',
          'opencv-python',
          'imutils',
          'PyWavelets',
          'wget',
          'sympy',
          'imageio',
          'astropy',
      ],
      packages=find_packages(),
      zip_safe=False)



