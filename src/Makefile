CFLAGS = $(shell python-config --cflags)
CFLAGS += -Ofast -march=native -mtune=native -ftree-vectorize
LDFLAGS = $(shell python-config --ldflags)
LDFLAGS += -fopenmp
CPPFLAGS = -I/usr/include/python3.8/

all: cython
	$(CC) $(CFLAGS) $(LDFLAGS) $(CPPFLAGS) -o pam *.c

cython:
	cythonize pam.pyx

cpp : cython

.PHONY: clean
clean:
	$(RM) -rf build pam.c *x86_64-linux-gnu.so
