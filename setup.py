from setuptools import setup, find_packages

def readme():
    with open("README.md") as f:
        return f.read()

setup(name='spyrit',
      version='1.0.0',
      description='Demo package',
      url='https://gitlab.in2p3.fr/antonio-tomas.lorente-mur/spyrit',
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
      install_requires=[
          'numpy (==1.19.3)',
          'matplotlib (==3.3.3)',
          'scipy (==1.5.2)',
          'torch (==1.7.1)',
          'torchvision (==0.8.2)',
          'Pillow (==7.2.0)',
          'opencv-python (==4.5.1.48)',
          'imutils (==0.5.3)',
          'PyWavelets',
          'imageio (==2.9.0)',
          'fht (==1.0.2)',
      ],
      packages=find_packages(),
      zip_safe=False)


#

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
