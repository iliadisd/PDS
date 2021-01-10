# PDS Lab2
### Ηλιάδης-Αποστολίδης Δημοσθένης 8811
Parallel and Distributed Systems

## kNN algorithm with parallel implementation (MPI).

## How to test
### Note : skip this if you have no problems with code.tar.gz and go directly to compile and run the testers. Even so, use the testers included.
The first time you try to run it, open two terminals. In the first one start a container to docker
and navigate to the code directory. In the second one just navigate regularly to the code directory.

In Docker :
<pre>
cd knnring
sudo make
cd ..
</pre>

Then in the other terminal, which must be opened in the folder of the code but not inside knnring.
(say 8811_Lab2).

<pre>
sudo tar -czvf code.tar.gz knnring
</pre>

And lastly to compile and run the testers, in Docker (inside 8811_Lab2) :

<pre>
sudo make //This compiles and runs automatically the sequential test.
make test_mpi //This compiles and runs automatically the mpi test for 12.
</pre>

After that you can run :
<pre>
./test_sequential //For sequential
mpirun -np <number_of_processes> ./test_mpi //For mpi, and to select
</pre>


Note :
1) To run different data, change tester.c lines 21 to 24 for sequential and lines 192 to 194
of test_mpi. In test_mpi please enter n = n/p. The tester gets p
by giving the number of processes via mpirun. Then in Docker (in 8811_Lab2) :
<pre>
sudo make clean
sudo make
make test_mpi
</pre>

2) Sometimes, mpi might look stuck while validating. This is not of course not counted
in the shown time.

3) Attributes are :
<pre>
Corel Image Features Data Set
{
n = 68040
ColorHistogram d = 32
ColorMoments d = 9
CoocTexture d = 16
LayoutHistory d = 32
}

TV News Channel Commercial Detection Data Set
{
d = 17
BBC n=17720
CNN n = 22545
CNNIBN n = 33117
NDTV n = 17051
TIMESNOW n = 39252
}

MiniBooNE particle identification Data Set
n = 130065 d = 50

FMA: A Dataset For Music Analysis Data Set
n = 106574 d = 518
</pre>

That's all.
