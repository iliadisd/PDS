SHELL := /bin/bash

CC = sm_30

NVCC	= nvcc -arch=$(CC)
SAMPLE	= global_kernel
ARGS	= -ptx

all: sample

sample:
	$(NVCC) $(ARGS) $(SAMPLE).cu

clean:
	rm -rf *~ *.ptx
