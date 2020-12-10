from setuptools import setup

setup(name='spyrit',
      version='0.1',
      description='Single-Pixel Image Reconstruction Toolbox for Python ',
      url='https://gitlab.in2p3.fr/antonio-tomas.lorente-mur/spyrit',
      author='Antonio Tomas Lorente Mur, Nicolas Ducros, Sebastien Crombez',
      author_email='Nicolas.Ducros@insa-lyon.fr',
      license='Attribution-ShareAlike 4.0 International',
      install_requires=[
          'numpy (>1.3.0)',
          'matplotlib (>2.2.4)',
          'scipy (>1.1.0)',
          'torch (>1.1.0)',
          'torchvision (>0.2.2)',
          'PIL (>5.3.0)',
          'cv2 (>4.0.0)',
          'imutils (>0.5.3)',
          'pywt (>1.0.1)',
      ],
      dependency_links=['https://github.com/nbarbey/fht'],
      packages=['spyrit'],
      zip_safe=False)
