import os

from setuptools import dist
dist.Distribution().fetch_build_eggs(['Cython>=0.15.1', 'numpy>=1.10'])

from distutils.core import setup, Extension
from Cython.Build import cythonize

os.environ['CFLAGS'] = '-Wall -std=c99 -O3 -march=native -mtune=native -ftree-vectorize'
os.environ['LDFLAGS'] = '-fopenmp -lm'

setup(
    ext_modules=cythonize(Extension(
        name="pam",
        sources = ["pam.pyx"])
))
