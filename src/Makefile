CFLAGS = $(shell python-config --cflags)
CFLAGS += "-O3 -march=native -mtune=native -ftree-vectorize"
LDFLAGS = $(shell python-config --ldflags)
LDFLAGS += "-fopenmp"

all: cython
	$(CC) $(CFLAGS) $(LDFLAGS) -o pam *.c

cython:
	cythonize pam.pyx

cpp : cython

.PHONY: clean
clean:
	$(RM) -rf build pam.c *x86_64-linux-gnu.so
