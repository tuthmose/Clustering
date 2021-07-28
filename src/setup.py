import os

from setuptools import dist
dist.Distribution().fetch_build_eggs(['Cython>=0.15.1', 'numpy>=1.10'])

from distutils.core import setup, Extension
from Cython.Build import cythonize

#python3 setup.py build_ext --inplace

os.environ['CFLAGS'] = '-Wall -std=c99 -O3 -march=native -mtune=native -ftree-vectorize'
os.environ['LDFLAGS'] = '-fopenmp -lm'

extensions = [
    Extension("pam",["pam.pyx"],
        #include_dirs=[...],
        #libraries=[...],
        #library_dirs=[...],
        ),
    Extension("scores",["scores.pyx"])]

setup(
    ext_modules=cythonize(extensions),
    )
