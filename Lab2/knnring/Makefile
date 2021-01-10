CC = gcc-7

MPICC = mpicc
MPIRUN = mpirun -np 4

CFLAGS = -O3
LDFLAGS = -lm -lopenblas

RM = rm -rf

.PHONY: lib

lib:
	cd src; $(CC) -c knnring_sequential.c $(CFLAGS); cd ..
	cd src; $(MPICC) -c knnring_mpi.c $(CFLAGS); cd ..

	cd src; ar rcs ../lib/knnring_sequential.a knnring_sequential.o; cd ..
	cd src; ar rcs ../lib/knnring_mpi.a knnring_mpi.o; cd ..
	
clean:
	$(RM) src/*.o lib/*.a $(EXECS)
