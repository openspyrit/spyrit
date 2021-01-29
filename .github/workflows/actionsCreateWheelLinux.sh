#!/bin/bash

set -e -x
echo ${PYTHONFOLDER}
mkdir /software
cd /home/
/opt/python/${PYTHONFOLDER}/bin/pip install numpy
/opt/python/${PYTHONFOLDER}/bin/python setup.py sdist bdist_wheel
auditwheel repair /home/dist/*.whl -w /software/wheelhouse/ --plat "manylinux2014_x86_64"
cp -r /software/wheelhouse /home/

