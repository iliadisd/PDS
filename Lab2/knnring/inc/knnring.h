#ifndef KNN_H
#define KNN_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <math.h>
#include <float.h>
#include <time.h>
#include <cblas.h>

#include <sys/time.h>
#include <sys/times.h>

typedef struct knnresult{
    int    * nidx;
    double * ndist;
    int      m;
    int      k;
}knnresult;

knnresult kNN(double * X, double * Y, int n, int m, int d, int k);

knnresult distrAllkNN(double * X, int n, int d, int k);

void swap_double(double* a, double* b);
void swap_int(int* a, int* b);

int part(double* arr, int* idx, int l, int r);

double quickselect(double* arr, int* idx_list, int l, int r, int k, int *idx);

#endif
