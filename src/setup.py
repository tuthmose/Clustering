import os
import sys
import subprocess

from setuptools import dist
from distutils.core import setup, Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext

args = sys.argv[1:]

if "clean" in args:
    print("Deleting Cython files")
    subprocess.Popen("rm -rf build", shell=True, executable="/bin/bash")
    subprocess.Popen("rm -rf __pycache__", shell=True, executable="/bin/bash")
    subprocess.Popen("rm -rf *.c", shell=True, executable="/bin/bash")
    subprocess.Popen("rm -rf *.so", shell=True, executable="/bin/bash")    
elif "build_ext" in args:
    dist.Distribution().fetch_build_eggs(['Cython>=0.15.1', 'numpy>=1.10'])
    #python3 setup.py build_ext --inplace
    os.environ['CFLAGS'] = '-Wall -std=c99 -O3 -march=native -mtune=native -ftree-vectorize'
    os.environ['LDFLAGS'] = '-fopenmp -lm'
    os.environ['ARCHFLAGS'] = "-arch x86_64"
    
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
