CC=gcc
CFLAGS= -std=c99 -fPIC
<<<<<<< HEAD
OPTFLAGS=-O3 -ffast-math -ftree-vectorize -march=native -fomit-frame-pointer
CPPFLAGS=-I. -I/usr/include/python3.7/ -I/usr/local/lib/python3.7/dist-packages/numpy/core/include/
=======
OPTFLAGS=-O3 -ffast-math -ftree-vectorize -march=native -fomit-frame-pointer -finline-functions
#CPPFLAGS=-I. -I/usr/include/python3.6/ -I/usr/local/lib/python3.6/dist-packages/numpy/core/include/
CPPFLAGS=-I. -I/usr/include/python3.7m/ -I/usr/local/lib/python3.7/dist-packages/numpy/core/include/
>>>>>>> test
LDFLAGS=-lm -fopenmp -lgsl -lgslcblas -lblas
INCLUDE = usr.h
SHELL	= /bin/sh
OBJECTS  = usr.o

SWIG     = swig 
SWIGOPT  = -python -Wall -l/usr/local/lib/python3.7/dist-packages/numpy/numpy.i
<<<<<<< HEAD
=======
#SWIGOPT  = -python -Wall -l/usr/local/lib/python3.6/dist-packages/numpy/numpy.i
>>>>>>> test
SWIGOBJS = pyusr_wrap.o

all: usr.exe pyusr

pyusr: $(OBJECTS) $(SWIGOBJS)
	$(CC) -shared -o _pyusr.so $(OBJECTS) $(SWIGOBJS) $(LDFLAGS)
	
pyusr_wrap.c: pyusr.i
	$(SWIG) $(SWIGOPT) pyusr.i
	
usr.exe: usr.o
	$(CC) -o usr.exe usr.o $(LDFLAGS)
	
clean:
	rm -f *.so *.o pyusr.py *.pyc *.a *.gch pyusr_wrap.c usr_wrap.c
	
.SUFFIXES : .c .h .o

.c.o:
	$(CC) $(INCLUDE) $(CFLAGS) $(OPTFLAGS) $(CPPFLAGS) -c $*.c
