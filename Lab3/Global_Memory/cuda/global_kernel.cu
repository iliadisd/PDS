#include <stdio.h>
#include <math.h>
#define INPUT(i,j) A[(i) + (m)*(j)]
#define OUTPUT(i,j) B[(i) + (m)*(j)]
#define H(i,j) Gauss[(i) + (Reg_x)*(j)]

__global__ void global_kernel(float const* const A, float *B, float const* const Gauss, int const m, int const n, int const threads, float const s, int const Reg_x, int const Reg_y)
{
// Coordinates of threads and blocks and initialization.
	int k = blockIdx.x;
  int l = blockIdx.y;
	int a,b,c,d,e,f;
	int xx, yy, xk, yk, xxk, yyk;
	float w,w1,z1;
	float norm=0.0;
	float Z = 0;
	float sub;
	if(k<1 && l<1)
	{
		w1 = m/threads ;
		z1 = n/threads ;
		for(e=0 ; e<w1 ; e++)
		{
			for(f=0 ; f<z1 ; f++)
			{
				Z = 0;
				//Coordinates of the centers of each element.
				xx = threadIdx.x*w1 + e;
				yy = threadIdx.y*z1 + f;
				for(a=0; a<m ; a++)
				{
					for(b=0; b<n ; b++)
					{
						norm = 0.0;
						for(c=0 ; c<Reg_x ; c++)
						{
							for(d=0 ; d<Reg_y ; d++)
							{
								//Neighboors of the pixel examined.
								xk = xx - Reg_x/2 + c;
								yk = yy - Reg_y/2 + d;
								//Coordinates of these neighboors.
								xxk = a - Reg_x/2 + c;
								yyk = b - Reg_y/2 + d;
								sub = INPUT(xk,yk) - INPUT(xxk,yyk);
								sub = H(c,d)*sub;
								sub = powf(sub,2);
								norm += sub;
							}
						}
						Z += expf(-norm/s);
						w = expf(-norm/s) ;
						OUTPUT(xx,yy) += w*INPUT(a,b);
					}
				}
				OUTPUT(xx,yy) = (OUTPUT(xx,yy)/Z);
				__syncthreads();
			}
		}
	}
}
