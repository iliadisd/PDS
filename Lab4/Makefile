CC=gcc
CFLAGS=-O3

default: all

BMM_openmp:
	$(CC) $(CFLAGS) -o BMM_openmp BMM_openmp.c -fopenmp

BMM_seq:
	$(CC) $(CFLAGS) -o BMM_seq BMM_seq.c


.PHONY: clean

all: BMM_openmp BMM_seq
clean:
	rm -f BMM_openmp BMM_seq 
