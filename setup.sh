#!/bin/bash

#numpy and other stuff
apt-get install build-essential python-setuptools python-numpy python-scipy python-dev python-pip python-nose g++ libopenblas-dev git libatlas-dev libatlas3-base

#PyYAML
pip install pyyaml

#Matplotlib
apt-get build-dep python-matplotlib

#Theano
pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

#CUDA
apt-get install nvidia-cuda-toolkit

#scikit-learn
apt-get install python-sklearn

#Pylearn2
git clone git://github.com/lisa-lab/pylearn2.git
export PYLEARN2_DATA_PATH=${PWD}/pylearn2
