#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#define ARRAY_SIZE 1000

int main(){


int n = ARRAY_SIZE;
int i, j, k;
int A[n][n];
int c3[n];

//srand(0);
for (i = 1; i<=n; ++i){
  c3[i] = 0;
    for(j = 1; j <= n; ++j){
      A[i][j] = rand() % 2;
      }
    }

for (i = 1; i <= n; ++i){
  for (j = 1; j <= n; ++j){
    if (A[i][j] == 1){
      A[j][i] = 1;
    } else if(A[j][i] == 1){
      A[i][j] = 1;
    }
      printf("%d\t",A[i][j]);
    }
    printf("\n");
}

clock_t start, end;
double time_used;
start = clock();

for(i = 1; i <= n-2 ; ++i) {
  for(j = 1; j <= n-1 ; ++j) {
    for(k = 1; k <= n ; ++k)  {
      if (i != j && i != k && A[i][j] == A[j][k] == A[k][i] == 1) {
        c3[i]++;
        c3[j]++;
        c3[k]++;
      }
    }
  }
}

end = clock();

for(k = 1; k <= n ; ++k){
  printf("%d\n\n",c3[k]);

}

time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
printf ("V1 took %f seconds to execute. \n", time_used);
}
