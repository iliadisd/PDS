 #include <stdio.h>
 #include <stdlib.h>
 #include <omp.h>
 #include "mmio.h"
 #include "mmio.c"
 #include <stdint.h>
 #include <time.h>
 #include <unistd.h>
 double BILLION =1000000000;
 struct timespec start, stop;
 double accum;
 uint32_t M, N,n, nz, i, j, k, *coo_col, *coo_row;
 int NUM_THREADS;


 uint32_t *Multiply(uint32_t *coo_row, uint32_t *coo_col, uint32_t n, uint32_t nz){
   uint32_t * A_csc_row = (uint32_t *)malloc(nz     * sizeof(uint32_t));
   uint32_t * A_csc_col = (uint32_t *)malloc((n + 1) * sizeof(uint32_t));
   uint32_t * B_csc_row = (uint32_t *)malloc(nz     * sizeof(uint32_t));
   uint32_t * B_csc_col = (uint32_t *)malloc((n + 1) * sizeof(uint32_t));
   uint32_t * C_csc_row = (uint32_t *)malloc(nz     * sizeof(uint32_t));
   uint32_t * C_csc_col = (uint32_t *)malloc((n + 1) * sizeof(uint32_t));
   uint32_t isOneBased = 0;

   coo2csc(A_csc_row, A_csc_col,
           coo_row, coo_col,
           nz, n,
           isOneBased);

   coo2csc(B_csc_row, B_csc_col,
                   coo_row, coo_col,
                   nz, n,
                   isOneBased);

 printf("\n");

   if( clock_gettime( CLOCK_REALTIME, &start) == -1 ) {
      perror( "clock gettime" );
      exit( EXIT_FAILURE );
    }

    C_csc_col[0] = 0;
        uint32_t counter = 0;
        #pragma omp parallel num_threads(NUM_THREADS) private (i,j,k)
        for(uint32_t i=1;i<n+1;i++){
          #pragma omp for nowait
          for(uint32_t j=A_csc_col[i-1];j<A_csc_col[i];j++){
            for (uint32_t k = B_csc_col[A_csc_row[j]]; k < B_csc_col[A_csc_row[j] + 1]; k++) {
              if (B_csc_row[k] == i-1) {
                #pragma omp critical
                {
                C_csc_row[i-1] = B_csc_row[k];
                counter++;
                C_csc_col[i] = counter;
                }
                #pragma omp cancel for
              }
            }
          }
          if (C_csc_col[i] == 0){
            C_csc_col[i] = C_csc_col[i-1];
          }
        }

   if( clock_gettime( CLOCK_REALTIME, &stop) == -1 ) {
       perror( "clock gettime" );
       exit( EXIT_FAILURE );
     }
/*
     for (uint32_t i = 0; i < n+1; i++){
       printf("%d\t", C_csc_col[i]);
     }
/*
   printf("\n\n");
     for (uint32_t i = 0; i < nz; i++){
       printf("%d\t", C_csc_row[i]);
     }
*/

   printf("\n%d\n", counter);

   accum = (double)(( stop.tv_sec - start.tv_sec )
              + ( stop.tv_nsec - start.tv_nsec )
                / BILLION);
   printf( "BMM_OpenMP took %lf seconds to execute.\n", accum );

   free( A_csc_row );
   free( A_csc_col );
   free( B_csc_row );
   free( B_csc_col );
   free( C_csc_row );
   free( C_csc_col );
   free( coo_col );
   free( coo_row );
 }

void coo2csc(
  uint32_t       * const row,
  uint32_t       * const col,
  uint32_t const * const row_coo,
  uint32_t const * const col_coo,
  uint32_t const         nz,
  uint32_t const         n,
  uint32_t const         isOneBased
) {
  for (uint32_t l = 0; l < n+1; l++) col[l] = 0;

  for (uint32_t l = 0; l < nz; l++)
    col[col_coo[l] - isOneBased]++;

  for (uint32_t i = 0, cumsum = 0; i < n; i++) {
    uint32_t temp = col[i];
    col[i] = cumsum;
    cumsum += temp;
  }
  col[n] = nz;

  for (uint32_t l = 0; l < nz; l++) {
    uint32_t col_l;
    col_l = col_coo[l] - isOneBased;

    uint32_t dst = col[col_l];
    row[dst] = row_coo[l] - isOneBased;

    col[col_l]++;
  }

  for (uint32_t i = 0, last = 0; i < n; i++) {
    uint32_t temp = col[i];
    col[i] = last;
    last = temp;
  }

}

int main(int argc, char *argv[]) {

  int ret_code;
  MM_typecode matcode;
  FILE *f;

  double *val;

  if (argc < 2)
{
  fprintf(stderr, "Usage: %s [martix-market-filename]\n", argv[0]);
  exit(1);
}
  else
  {
      if ((f = fopen(argv[1], "r")) == NULL)
          exit(1);
  }

  if (mm_read_banner(f, &matcode) != 0)
  {
      printf("Could not process Matrix Market banner.\n");
      exit(1);
  }

  if (mm_is_complex(matcode) && mm_is_matrix(matcode) &&
          mm_is_sparse(matcode) )
  {
      printf("Sorry, this application does not support ");
      printf("Market Market type: [%s]\n", mm_typecode_to_str(matcode));
      exit(1);
  }

  if ((ret_code = mm_read_mtx_crd_size(f, &M, &N, &nz)) !=0)
      exit(1);


  coo_col = (uint32_t *) malloc(nz * sizeof(uint32_t));
  coo_row = (uint32_t *) malloc(nz * sizeof(uint32_t));
  val = (double *) malloc(nz * sizeof(double));


  if (!mm_is_pattern(matcode))
  {
  for (i=0; i<nz; i++)
  {
      fscanf(f, "%d %d %lg\n", &coo_row[i], &coo_col[i], &val[i]);
      coo_row[i]--;  /* adjust from 1-based to 0-based */
      coo_col[i]--;
  }
  }
  else
  {
  for (i=0; i<nz; i++)
  {
      fscanf(f, "%d %d\n", &coo_row[i], &coo_col[i]);
      val[i]=1;
      coo_row[i]--;  /* adjust from 1-based to 0-based */
      coo_col[i]--;
  }
  }

  if (f !=stdin) fclose(f);
  mm_write_banner(stdout, matcode);
  mm_write_mtx_crd_size(stdout, M, N, nz);


  n = M;

  free(val);

  do {
       printf("Enter the number of threads :" );
       scanf("%d", &NUM_THREADS);
   } while (NUM_THREADS <= 0);

  Multiply(coo_row, coo_col, n, nz);

return 0;
}
