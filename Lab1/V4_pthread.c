 #include <stdio.h>
 #include <stdlib.h>
 #include "mmio.h"
 #include "mmio.c"
 #include <stdint.h>
 #include <time.h>
 #include <unistd.h>
 #include <pthread.h>
 double MILLION = 1000000;
 double BILLION = 1000000000;
 struct timespec start, stop;
 double accum;
 uint32_t n;
 uint32_t nz;
 int NUM_THREADS;
 long t;
 //pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;

 typedef enum { F, T } bool;

bool nb=F; //neighboor hadamard
uint32_t * csc_row ;
uint32_t * csc_col;
uint32_t * c3;

 uint32_t *vertexWiseTriangleCounts(){

int incr = (int)((n/NUM_THREADS) * (int)t);
int incr2 = (int)((n/NUM_THREADS) * ((int)t+1));

if (t == 0){
 incr = incr+1;
}else if (t == NUM_THREADS-1){
  incr2 = n;
}

    for(uint32_t i=incr;i<incr2;i++){
       for(uint32_t j=1;j<n+1;j++){
         nb=F; //neighboor hadamard
    		   for(uint32_t k=csc_col[i-1];k<csc_col[i];k++){
             if (csc_row[k]==j-1){
               nb = T;
             }
             if (nb==T){
    			   for(uint32_t l=csc_col[j-1];l<csc_col[j];l++){
              if (csc_row[k]==csc_row[l]){
                c3[csc_row[k]]++;
                c3[i-1]++;
                c3[j-1]++;
              }
             }
             }
            }
         }
      }

      printf ("%ld thread is done. Waiting to join... \n", t);
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
uint32_t M, N, i, j, k, *coo_col, *coo_row;
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

  csc_row = (uint32_t *)malloc(nz     * sizeof(uint32_t));
  csc_col = (uint32_t *)malloc((n + 1) * sizeof(uint32_t));


  uint32_t isOneBased = 0;

  coo2csc(csc_row, csc_col,
          coo_row, coo_col,
          nz, n,
          isOneBased);

c3 = (uint32_t *)malloc(n * sizeof(uint32_t));

 for (uint32_t i=0;i<n;i++){
   c3[i]=0;
 }

  do {
       printf("Enter the number of threads :" );
       scanf("%d", &NUM_THREADS);
   } while (NUM_THREADS <= 0);

   pthread_t threads[NUM_THREADS];
   int rc;

   if( clock_gettime( CLOCK_REALTIME, &start) == -1 ) {
      perror( "clock gettime" );
      exit( EXIT_FAILURE );
    }

   for(t=0;t<NUM_THREADS;t++){
     printf("In main: creating thread %ld\n", t);
     rc = pthread_create(&threads[t], NULL, (void*)vertexWiseTriangleCounts, NULL);
     usleep(100000);
     if (rc){
       printf("ERROR; return code from pthread_create() is %d\n", rc);
       exit(-1);
       }
     }
     for (int i = 0; i < NUM_THREADS; i++)
        pthread_join(threads[i], NULL);

  if( clock_gettime( CLOCK_REALTIME, &stop) == -1 ) {
      perror( "clock gettime" );
      exit( EXIT_FAILURE );
    }

  accum = (double)(( stop.tv_sec - start.tv_sec )
                 + ( stop.tv_nsec - start.tv_nsec )
                     / BILLION) ;
 double delay = (double)(NUM_THREADS*100000 /MILLION );

 printf( "V4 took %lf seconds to execute.\n", accum-delay);

   free( csc_row );
   free( csc_col );
   free(coo_col);
   free(coo_row);
   free(c3);
return 0;
}
