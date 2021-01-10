SHELL := /bin/bash
CC = gcc-7
MPICC = mpicc
MPIRUN = mpirun -np 12

test_sequential:
	tar -xvzf code.tar.gz
	cd knnring; make lib; cd ..
	cd knnring; cp lib/*.a inc/knnring.h ../; cd ..
	$(CC) tester.c knnring_sequential.a -o $@ -lm -lopenblas
	./test_sequential


test_mpi:
	tar -xvzf code.tar.gz
	cd knnring; make lib; cd ..
	cd knnring; cp lib/*.a inc/knnring.h ../; cd ..
	$(MPICC) tester_mpi.c knnring_mpi.a -o $@ -lm -lopenblas
	$(MPIRUN) ./test_mpi

clean:
	rm test_sequential test_mpi
