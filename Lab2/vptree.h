#ifndef VPTREE_H
#define VPTREE_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <math.h>
#include <float.h>
#include <time.h>
#include <pthread.h>

#include <sys/time.h>
#include <sys/times.h>

/*
*************************************************
*    @file   vptree.h                           *
*    @author amanolis <amanolis@ece.auth.gr>    *
*    @date   Mon Oct 21 22:28:00 2019           *
*    @brief  Main utility functions             *
*************************************************
*/

typedef struct vptree vptree;
typedef struct dist_thread dist_thread;
typedef struct tree_thread tree_thread;

struct vptree
{
    double md, *vp;
    int idx;
    vptree *inner, *outer;
};

struct dist_thread
{
    int n, d, start, end;
    double *X, *dist;
};

struct tree_thread
{
    int n, d, *idx;
    double *X;
    vptree *T;
};

/*
**************************************************************
*    Build vantage-point tree given input dataset X          *
*    param X: Input data points, stored as [n-by-d] array    *
*    param n: Number of data points (rows of X)              *
*    param d: Number of dimensions (columns of X)            *
*    return The vantage-point tree                           *
**************************************************************
*/

vptree * buildvp(double *X, int n, int d);

/*
**************************************************************
*    Build vantage-point tree given input dataset X          *
*    param X: Input data points, stored as [n-by-d] array    *
*    param n: Number of data points (rows of X)              *
*    param d: Number of dimensions (columns of X)            *
*    return The vantage-point tree                           *
**************************************************************
*/

vptree * buildvp_recursive(double *X, int n, int d, int *idx_list);

/*
****************************************************************
*    Return vantage-point subtree with points inside radius    *
*    param node: A vantage-point tree                          *
*    return The vantage-point subtree                          *
****************************************************************
*/

vptree * getInner(vptree * T);

/*
*****************************************************************
*    Return vantage-point subtree with points outside radius    *
*    param node: A vantage-point tree                           *
*    return The vantage-point subtree                           *
*****************************************************************
*/

vptree * getOuter(vptree * T);

/*
*****************************************************
*    Return median of distances to vantage point    *
*    param node: A vantage-point tree               *
*    return The median distance                     *
*****************************************************
*/

double getMD(vptree * T);

/*
*******************************************************
*    Return the coordinates of the vantage point      *
*    param node: A vantage-point tree                 *
*    return The coordinates [d-dimensional vector]    *
*******************************************************
*/

double * getVP(vptree * T);

/*
*************************************************************
*    Return the index of the vantage point                  *
*    param node: A vantage-point tree                       *
*    return The index to the input vector of data points    *
*************************************************************
*/

int getIDX(vptree * T);

/*
***************************************
*    Function that swaps 2 numbers    *
***************************************
*/

void swap(double* a, double* b);

/*
********************************************************************
*    This function takes last element as pivot, places the pivot   *
*    element at its correct position in sorted array, and places   *
*    all smaller (smaller than pivot) to left of pivot and all     *
*    greater elements to right of pivot.                           *
********************************************************************
*/

int partition(double* arr, int l, int r);

/*
************************************
*    Quickselect implementation    *
************************************
*/

double quickselect(double* arr, int l, int r, int k);

/*
*****************************
*    Free Memory of Tree    *
*****************************
*/

void destroy(vptree *T);

#endif
