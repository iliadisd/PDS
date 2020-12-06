#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#define ARRAY_SIZE 12

int main(){


int n = ARRAY_SIZE;
int i, j, k;
int A[n][n];
int c3[n];

srand(time(NULL));
for (i = 0; i<n; ++i){
  c3[i] = 0;
    for(j = 0; j < n; ++j){
        A[i][j] = rand() % 2;
      }
    }

for (i = 0; i < n; ++i){
  for (j = 0; j < n; ++j){
    if (A[i][j] == 1){
      A[j][i] = 1;
    } else if(A[j][i] == 1){
      A[i][j] = 1;
    }
    if (i==j){
      A[i][j] = 0;
    }
      printf("%d\t",A[i][j]);
    }
    printf("\n");
}

clock_t start, end;
double time_used;
start = clock();

for(i = 0; i < n-2 ; ++i) {
  for(j = i + 1; j < n-1 ; ++j) {
    for(k = j + 1; k < n ; ++k)  {
      if ((A[i][j]==1)&&(A[i][k]==1)&&(A[j][k]==1)){
        c3[i]++;
        c3[j]++;
        c3[k]++;
      }
    }
  }
}

end = clock();
printf("\n");
for(k = 0; k < n ; ++k){
  printf("%d\t%d\n",k,c3[k]);
}

time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
printf ("V2 took %f seconds to execute. \n", time_used);

return 0;

}
