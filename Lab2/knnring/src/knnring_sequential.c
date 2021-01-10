#include "../inc/knnring.h"

double BILLION =1000000000;
struct timespec start, stop;
double accum;
typedef struct knnresult knnresult;

knnresult kNN(double * X, double * Y, int n, int m, int d, int k)
{
    knnresult knnres;

    knnres.m = m;
    knnres.k = k;

    double *dist     = malloc(m*n * sizeof(double));
    double *dist_copy  = malloc(n   * sizeof(double));
    int    *idx_list = malloc(n   * sizeof(int));
    knnres.nidx      = malloc(m*k * sizeof(int));
    knnres.ndist     = malloc(m*k * sizeof(double));


    if( clock_gettime( CLOCK_REALTIME, &start) == -1 ) {
       perror( "clock gettime" );
       exit( EXIT_FAILURE );
     }

    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, n, m, d, -2, X, n, Y, m, 0, dist, n);

    double *X_2 = malloc(n * sizeof(double));
    for(int i=0; i<n; i++)
    {
        X_2[i] = 0.0;
        for(int j=0; j<d; j++)
            X_2[i] += X[j*n + i] * X[j*n + i];
    }

    double *Y_2 = malloc(m * sizeof(double));
    for(int i=0; i<m; i++)
    {
        Y_2[i] = 0.0;
        for(int j=0; j<d; j++)
            Y_2[i] += Y[j*m + i] * Y[j*m + i];
    }

    for(int i=0; i<n; i++)
    {
        for(int j=0; j<m; j++)
        {
            dist[j*n + i] += X_2[i] + Y_2[j];
            if( dist[j*n + i] < 1e-8 )
                dist[j*n + i] = 0.0;
            dist[j*n + i] = sqrt(dist[j*n + i]);
        }
    }

    for(int i=0; i<m; i++)
    {
        for(int j=0; j<n; j++)
        {
            dist_copy[j] = dist[j + i*n];
            idx_list[j] = j;
        }

        int index;
        knnres.ndist[m*(k-1) + i] = quickselect(dist_copy, idx_list, 0, n-1, k, &index);
        knnres.nidx[m*(k-1) + i] = idx_list[index];


        for(int z=k-2; z>=0; z--)
        {
            knnres.ndist[m*z + i] = quickselect(dist_copy, idx_list, 0, z, z+1, &index);
            knnres.nidx[m*z + i] = idx_list[index];
        }

    }

    if( clock_gettime( CLOCK_REALTIME, &stop) == -1 ) {
        perror( "clock gettime" );
        exit( EXIT_FAILURE );
      }

    accum = (double)(( stop.tv_sec - start.tv_sec )
               + ( stop.tv_nsec - start.tv_nsec )
                 / BILLION);
    printf( "knnring_sequential took %lf seconds to execute.\n", accum );

    free(dist);
    free(dist_copy);
    free(idx_list);

    return knnres;
}

void swap_int(int* a, int* b)
{
    int swap = *a;
    *a = *b;
    *b = swap;
}

void swap_double(double* a, double* b)
{
    double swap = *a;
    *a = *b;
    *b = swap;
}

int part(double* arr, int* idx_list, int l, int r)
{
    double x = *(arr + r);
    int i = l;
    for (int j = l; j <= r - 1; j++)
    {
        if (*(arr + j) <= x)
        {
            swap_double(&*(arr + i), &*(arr + j));
            swap_int(&*(idx_list + i), &*(idx_list + j));
            i++;
        }
    }
    swap_double(&*(arr + i), &*(arr + r));
    swap_int(&*(idx_list + i), &*(idx_list + r));
    return i;
}

double quickselect(double* arr, int* idx_list, int l, int r, int k, int *idx)
{
    if (k > 0 && k <= r - l + 1)
    {
        int index = part(arr, idx_list, l, r);
        if (index - l == k - 1)
        {
            *idx = index;
            return *(arr + index);
        }
        if (index - l > k - 1)
            return quickselect(arr, idx_list, l, index - 1, k, idx);
        return quickselect(arr, idx_list, index + 1, r, k - index + l - 1, idx);
    }
    return DBL_MAX;
}
