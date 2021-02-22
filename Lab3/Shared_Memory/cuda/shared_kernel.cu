#include <math.h>
#include <stdio.h>
#define INPUT(i,j) A[(i) + (m + 2*(Reg_x/2))*(j)]
#define OUTPUT(i,j) B[(i) + (m)*(j)]
#define H(i,j) Gauss[(i)*(Reg_y) + (j)]
#define threads 16
#define blocks 16
//Change the dimensions below to "play" with images and recompile.
#define m 256
#define n 256
__global__ void myKernel(float const* const A, float *B, float const* const Gauss, int const Reg_x, int const Reg_y, float const s)
{
  // Coordinates of threads and blocks and initialization.
  int k = blockIdx.x;
  int l = blockIdx.y;
  int x,y,xk,yk,xx,yy,xxk,yyk;
  int w1,w2,w3,c,d,e,f;
	 __shared__ float output[threads][threads];
	 __shared__ float z_per_block[threads][threads];
	 __shared__ float Z[m/blocks][n/blocks];
	 float w;
	 float z_per_thread=0.0;
	 float sub;
	 float norm=0.0;
  if( k<m && l<n) {
     // Div the threads with the elements to get the pixels of every thread.
	   w2 = m/gridDim.x;
	   w1 = w2*m/threads;
	   w3 = threads/w2;
	   z_per_thread=0.0;
	   if(threadIdx.x == 0 && threadIdx.y == 0)
	    {
		    for( e=0 ; e<w2 ; e++)
		      {
			      for( f=0 ; f<w2 ; f++)
			        {
				        Z[e][f] = 0;
			        }
		      }
		    for(e=0; e<threads; e++)
		      {
			     for(f=0 ; f<threads ; f++)
			        {
				      output[e][f] = 0;
			        }
		      }
	     }
	// Synchronize to make sure all tables above are zero.
	__syncthreads();
  //Coordinates of the centers of each element.
	x = blockIdx.x*(m/blocks) + threadIdx.x/w3 + Reg_x/2;
	y = blockIdx.y*(n/blocks) + threadIdx.y/w3 + Reg_y/2;
	/* Σε κάθε νήμα θα υπολογιστεί :
							   το 1/16 του pixel για  64χ64 ,δηλαδή ανά m/4,n/4
							   το 1/4 του pixel για 128χ128 ,δηλαδή ανά m/2,n/2
							   το 1/1 του pixel για 256χ256 ,δηλαδή ανά m/1,n/1 	*/
	for( e=0 ; e<(m/w3) ; e++)
	{
		for( f=0 ; f<(n/w3) ; f++)
		{
			norm = 0.0;
      //Coordinates of the centers of each element.
			xx = (threadIdx.x % w3)*w1 + e + Reg_x/2;
			yy = (threadIdx.y % w3)*w1 + f + Reg_y/2;
			for( c=0 ; c<Reg_x ; c++)
			{
				for( d=0 ; d<Reg_y ; d++)
				{
          //Neighboors of the pixel examined.
					xk = x - Reg_x/2 + c;
					yk = y - Reg_y/2 + d;
          //Coordinates of these neighboors.
					xxk = xx  - Reg_x/2 + c;
					yyk = yy  - Reg_y/2 + d;
					sub = INPUT(xk,yk) - INPUT(xxk,yyk);
					sub = H(c,d)*sub;
					sub = powf(sub,2);
					norm += sub ;
				}
			}
			w = expf(-norm/s);
			z_per_thread += expf(-norm/s);
			output[threadIdx.x][threadIdx.y] += w*INPUT(xx,yy);
		}
	}
	z_per_block[threadIdx.x][threadIdx.y] = z_per_thread;
	__syncthreads();
	if(threadIdx.x == 0 && threadIdx.y == 0)
	{
		for( e=0 ; e<threads ; e++)
		{
			for( f=0 ; f<threads ; f++)
			{
				Z[e/w3][f/w3] += z_per_block[e][f];
			}
		}
	}
	__syncthreads();
	if(threadIdx.x == 0 && threadIdx.y == 0)
	{
		for( e=0 ; e<threads ; e++)
		{
			for( f=0 ; f<threads ; f++)
			{
				OUTPUT(blockIdx.x*(m/blocks)+(e/w3) , blockIdx.y*(n/blocks)+(f/w3)) += (1/Z[e/w3][f/w3])*output[e][f];
			}
		}
	}
  }
}
