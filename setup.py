from setuptools import setup, find_packages, Extension
from os.path import join
import os
import sys
from sys import platform as _platform

def readme():
    with open("README.md") as f:
        return f.read()

# possible types
types = ("int", "long", "float", "double")

#To delay numpy import
class get_numpy_include(object):
    def __str__(self):
        import numpy
        return join(numpy.get_include(), 'numpy')

# generate sources
pth = os.getcwd()
template_file = join(os.getcwd(), "spyrit/fht/fht", "C_fht.template.c")
f = open(template_file, "r")
txt = f.read()
f.close()
for t in types:
    d = {"ctype":t}
    filled_txt = txt % d
    source = join(os.getcwd(), "spyrit/fht/fht", "C_fht_%(ctype)s.c" % d)
    f = open(source, "w")
    f.write(filled_txt)
    f.close()

# distutils

sys.path.extend('config_fc --fcompiler=gnu95 --f90flags=-fopenmp --f90exec=/usr/bin/gfortran '.split())

compile_args = '-fopenmp'
if _platform == "darwin":
  compile_args = '-Xpreprocessor ' + compile_args
if _platform == "win32":
  compile_args = '/openmp '

setup(name='spyrit',
      version='0.13.5',
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
      ext_modules=[Extension('fht._C_fht_%(ctype)s' % {"ctype":t},
                             [join('spyrit/fht/fht', 'C_fht_%(ctype)s.c') % {"ctype":t}],
                             include_dirs=[get_numpy_include()],
                             extra_compile_args=[compile_args],
                             extra_link_args=[compile_args],)
                   for t in types],
      install_requires=[
          'numpy (==1.19.3)',
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
      ],
      packages=find_packages(),
      zip_safe=False)



#
#      install_requires=[
#          'numpy',
#          'matplotlib',
#          'scipy',
#          'torch',
#          'torchvision',
#          'Pillow',
#          'opencv-python',
#          'imutils',
#          'PyWavelets',
#          'imageio',
#          'fht',
# 
