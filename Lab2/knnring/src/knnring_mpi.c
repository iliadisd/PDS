#include "../inc/knnring.h"
#include <mpi.h>
#include <stddef.h>

double BILLION =1000000000;
struct timespec start, stop;
double accum;

typedef struct knnresult knnresult;
typedef struct knnresult knnresult;

knnresult distrAllkNN(double * X, int n, int d, int k)
{
    MPI_Status send_status, recv_status;
    MPI_Request	send_request, recv_request;
    int p, id;
    MPI_Comm_rank(MPI_COMM_WORLD, &id);
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    knnresult knnres;
    double *Y    = (double *)malloc(n*d * sizeof(double));
    double *X_send = (double *)malloc(n*d * sizeof(double));
    double *X_recv = (double *)malloc(n*d * sizeof(double));
    double *dist = (double *)malloc(k * sizeof(double));
    int    *idx  =    (int *)malloc(k * sizeof(int));
    int source, dest;

    if(id == 0)
        source = p-1;
    else
        source = id-1;

    if(id == p-1)
        dest = 0;
    else
        dest = id+1;

    if( clock_gettime( CLOCK_REALTIME, &start) == -1 ) {
           perror( "clock gettime" );
           exit( EXIT_FAILURE );
     }


    memcpy(X_send, X, n*d*sizeof(double));
    MPI_Isend(X_send, n*d, MPI_DOUBLE, dest, 1, MPI_COMM_WORLD, &send_request);
    MPI_Irecv(X_recv, n*d, MPI_DOUBLE, source, 1, MPI_COMM_WORLD, &recv_request);
    memcpy(Y, X, n*d*sizeof(double));
    knnres = kNN(X, Y, n, n, d, k);
    int mul = id + p - 1;
    if(mul >= p)
        mul -= p;

    for(int i=0; i<n; i++)
        for(int z=0; z<k; z++)
            knnres.nidx[n*z + i] = knnres.nidx[n*z + i] + mul*n;
    for(int ip=0; ip<p-1; ip++)
    {
        MPI_Wait(&send_request, &send_status);
        MPI_Wait(&recv_request, &recv_status);

        memcpy(X, X_recv, n*d*sizeof(double));
        if(ip < p-2)
        {
            memcpy(X_send, X_recv, n*d*sizeof(double));
            MPI_Isend(X_send, n*d, MPI_DOUBLE, dest, 1, MPI_COMM_WORLD, &send_request);
            MPI_Irecv(X_recv, n*d, MPI_DOUBLE, source, 1, MPI_COMM_WORLD, &recv_request);
        }

        knnresult knn_temp;
        knn_temp = kNN(X, Y, n, n, d, k);

        mul--;
        if(mul < 0)
            mul = p-1;

        for(int i=0; i<n; i++)
            for(int z=0; z<k; z++)
                knn_temp.nidx[n*z + i] = knn_temp.nidx[n*z + i] + mul*n;
        for(int i=0; i<n; i++)
        {
            int z1 = 0, z2 = 0, z3 = 0;
            while (z1<k && z2<k && z3<k)
            {
                if (knnres.ndist[n*z1 + i] < knn_temp.ndist[n*z2 + i])
                {
                    dist[z3] = knnres.ndist[n*(z1) + i];
                    idx[z3++] = knnres.nidx[n*(z1++) + i];
                }else
                {
                    dist[z3] = knn_temp.ndist[n*(z2) + i];
                    idx[z3++] = knn_temp.nidx[n*(z2++) + i];
                }
            }
            while (z1 < k && z3<k)
            {
                dist[z3] = knnres.ndist[n*(z1) + i];
                idx[z3++] = knnres.nidx[n*(z1++) + i];
            }
            while (z2 < k && z3<k)
            {
                dist[z3] = knn_temp.ndist[n*(z2) + i];
                idx[z3++] = knn_temp.nidx[n*(z2++) + i];
            }
            for(int z=0; z<k; z++)
            {
                knnres.ndist[n*z + i] = dist[z];
                knnres.nidx[n*z + i] = idx[z];
            }
        }
    }



    free(Y);
    free(X_send);
    free(X_recv);
    free(dist);
    free(idx);

    double local_max = -DBL_MAX;
    double local_min = DBL_MAX;
    double global_max, global_min;

    for(int i=0; i<n ;i++)
    {
        if(knnres.ndist[n*(k-1) + i] > local_max)
            local_max = knnres.ndist[n*(k-1) + i];

        int counter = 0;
        while(knnres.ndist[n*counter + i] == 0.0)
            counter++;

        if(knnres.ndist[n*counter + i] < local_min)
            local_min = knnres.ndist[n*counter + i];
    }

    MPI_Reduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_min, &global_min, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);

    if(id == 0)
    {
       printf("Global max distance: %f \n", global_max);
       printf("Global min distance: %f \n", global_min);
       if( clock_gettime( CLOCK_REALTIME, &stop) == -1 ) {
           perror( "clock gettime" );
           exit( EXIT_FAILURE );
         }

       accum = (double)(( stop.tv_sec - start.tv_sec )
                  + ( stop.tv_nsec - start.tv_nsec )
                    / BILLION);
       printf( "knnring_MPI took %lf seconds to execute.\n", accum );
    }

    return knnres;
}

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

    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, n, m, d, -2, X, n, Y, m, 0, dist, n);

    double *X_2 = malloc(n * sizeof(double));
    for(int i=0; i<n; i++)
    {
        X_2[i] = 0.0;
        for(int z=0; z<d; z++)
            X_2[i] += X[n*z + i] * X[n*z + i];
    }
    double *Y_2 = malloc(m * sizeof(double));
    for(int i=0; i<m; i++)
    {
        Y_2[i] = 0.0;
        for(int z=0; z<d; z++)
            Y_2[i] += Y[m*z + i] * Y[m*z + i];
    }
    for(int i=0; i<n; i++)
    {
        for(int j=0; j<m; j++)
        {
            dist[n*j + i] += X_2[i] + Y_2[j];
            if( dist[n*j + i] < 1e-8 )
                dist[n*j + i] = 0.0;
            dist[n*j + i] = sqrt(dist[n*j + i]);
        }
    }
    for(int i=0; i<m; i++)
    {
        for(int j=0; j<n; j++)
        {
            dist_copy[j] = dist[n*i + j];
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
    free(dist);
    free(dist_copy);
    free(idx_list);
    free(X_2);
    free(Y_2);

    return knnres;
}

void swap_double(double* a, double* b)
{
    double swap = *a;
    *a = *b;
    *b = swap;
}

void swap_int(int* a, int* b)
{
    int swap = *a;
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
