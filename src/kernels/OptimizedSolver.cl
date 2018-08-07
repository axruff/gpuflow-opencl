/* These macros will be defined dynamically during building the program

#define TILE_SIZE_X		32
#define TILE_SIZE_Y		16

*/

#define BX 1
#define BY 1
#define IND(X, Y) ((Y) * pitch + (X))

__kernel void OptimizedSolver(
	__global	const	float*	d_img_1,	//  0 in     : 1st image 
	__global	const	float*	d_img_2,	//  1 in     : 2nd image
	__global	const	float*	du,			//  2 in	 : x-component of flow increment
	__global	const	float*	dv,			//  3 in	 : y-component of flow increment
	__global	const	float*	u,			//  4 in	 : x-component of flow field
	__global	const	float*	v,			//  5 in	 : y-component of flow field
						float	hx,			//  6 in     : grid spacing in x-direction
						float	hy,			//  7 in     : grid spacing in y-direction
						float	alpha,		//  8 in     : smoothness weight
						float	omega,		//  9 in     : sor overrelaxation parameter
						int		width,		// 10 in     : image width
						int		height,		// 11 in     : image height
						int		pitch,		// 12 in     : image pitch
	__global			float*	du_r,		// 13 out	 : du result
	__global			float*	dv_r		// 14 out	 : dv result
	)
{
	__local float l_img_1[TILE_SIZE_Y + 2 * BY][TILE_SIZE_X + 2 * BX];
	__local float l_img_2[TILE_SIZE_Y + 2 * BY][TILE_SIZE_X + 2 * BX];
	__local float	 l_du[TILE_SIZE_Y + 2 * BY][TILE_SIZE_X + 2 * BX];
	__local float	 l_dv[TILE_SIZE_Y + 2 * BY][TILE_SIZE_X + 2 * BX];
	__local float	  l_u[TILE_SIZE_Y + 2 * BY][TILE_SIZE_X + 2 * BX];
	__local float	  l_v[TILE_SIZE_Y + 2 * BY][TILE_SIZE_X + 2 * BX];

	size_t x = get_global_id(0);
	size_t y = get_global_id(1);
	size_t lx = get_local_id(0);
	size_t ly = get_local_id(1);

	if (x >= width || y >= height) {
		return;
	}

	// fill main area
	l_img_1[ly + BY][lx + BX] = d_img_1[IND(x, y)];
	l_img_2[ly + BY][lx + BX] = d_img_2[IND(x, y)];
	   l_du[ly + BY][lx + BX] =		 du[IND(x, y)];
	   l_dv[ly + BY][lx + BX] =      dv[IND(x, y)];
		l_u[ly + BY][lx + BX] =       u[IND(x, y)];
		l_v[ly + BY][lx + BX] =	      v[IND(x, y)];

	// left edge
	if (lx == 0) {
		if(x > 1) {
			l_img_1[ly + BY][0] = d_img_1[IND(x - 1, y)];
			l_img_2[ly + BY][0] = d_img_2[IND(x - 1, y)];
			   l_du[ly + BY][0] =	   du[IND(x - 1, y)];
			   l_dv[ly + BY][0] =	   dv[IND(x - 1, y)];
			    l_u[ly + BY][0] =	    u[IND(x - 1, y)];
				l_v[ly + BY][0] =	    v[IND(x - 1, y)];
		} else {
			l_img_1[ly + BY][0] = d_img_1[IND(BX, y)];
			l_img_2[ly + BY][0] = d_img_2[IND(BX, y)];
		}
	}
	// right edge
	if (lx == TILE_SIZE_X - 1 || x == width - 1) {
		if (x < width - 1) {
			l_img_1[ly + BY][TILE_SIZE_X + BX] = d_img_1[IND(x + 1, y)];
			l_img_2[ly + BY][TILE_SIZE_X + BX] = d_img_2[IND(x + 1, y)];
			   l_du[ly + BY][TILE_SIZE_X + BX] =	  du[IND(x + 1, y)];
			   l_dv[ly + BY][TILE_SIZE_X + BX] =	  dv[IND(x + 1, y)];
		    	l_u[ly + BY][TILE_SIZE_X + BX] =	   u[IND(x + 1, y)];
			    l_v[ly + BY][TILE_SIZE_X + BX] =   	   v[IND(x + 1, y)];
		} else {
			l_img_1[ly + BY][lx + BX + 1] = d_img_1[IND(width - 2, y)];
			l_img_2[ly + BY][lx + BX + 1] = d_img_2[IND(width - 2, y)];
		}
	}
	// upper edge
	if (ly == 0) {
		if (y > 1) {
			l_img_1[0][lx + BX] = d_img_1[IND(x, y - 1)];
			l_img_2[0][lx + BX] = d_img_2[IND(x, y - 1)];
			   l_du[0][lx + BX] =	   du[IND(x, y - 1)];
			   l_dv[0][lx + BX] =	   dv[IND(x, y - 1)];
				l_u[0][lx + BX] =		u[IND(x, y - 1)];
				l_v[0][lx + BX] =		v[IND(x, y - 1)];
		} else {
			l_img_1[0][lx + BX] = d_img_1[IND(x, BY)];
			l_img_2[0][lx + BX] = d_img_2[IND(x, BY)];
		}
	}
	// bottom edge
	if (ly == TILE_SIZE_Y - 1 || y == height - 1) {
		if (y < height - 1) {
			l_img_1[TILE_SIZE_Y + BY][lx + BX] = d_img_1[IND(x, y + 1)];
			l_img_2[TILE_SIZE_Y + BY][lx + BX] = d_img_2[IND(x, y + 1)];
			   l_du[TILE_SIZE_Y + BY][lx + BX] =	  du[IND(x, y + 1)];
			   l_dv[TILE_SIZE_Y + BY][lx + BX] =	  dv[IND(x, y + 1)];
				l_u[TILE_SIZE_Y + BY][lx + BX] =	   u[IND(x, y + 1)];
				l_v[TILE_SIZE_Y + BY][lx + BX] =	   v[IND(x, y + 1)];
		} else {
			l_img_1[ly + BY + 1][lx + BX] = d_img_1[IND(x, height - 2)];
			l_img_2[ly + BY + 1][lx + BX] = d_img_2[IND(x, height - 2)];
		}
	}

	// synchronize warp
	barrier(CLK_LOCAL_MEM_FENCE);
	float hx_2 = alpha / (hx * hx);
	float hy_2 = alpha / (hy * hy);

	// Derivatives variables
	float fx = (l_img_1[ly + BY][lx + BX + 1] - l_img_1[ly + BY][lx + BX - 1] + l_img_2[ly + BY][lx + BX + 1] - l_img_2[ly + BY][lx + BX - 1]) / (4.f * hx);
	float fy = (l_img_1[ly + BY + 1][lx + BX] - l_img_1[ly + BY - 1][lx + BX] + l_img_2[ly + BY + 1][lx + BX] - l_img_2[ly + BY - 1][lx + BX]) / (4.f * hy);
	float ft = l_img_2[ly + BY][lx + BX] - l_img_1[ly + BY][lx + BX];

	float J11 = fx * fx;
	float J22 = fy * fy;
	float J12 = fx * fy;
	float J13 = fx * ft;
	float J23 = fy * ft;

	// Compute weights 
	float xp = (x < width - 1)	* hx_2;
	float xm = (x > 0)			* hx_2;
	float yp = (y < height - 1)	* hy_2;
	float ym = (y > 0)			* hy_2;
	float sum = (xp + xm + yp + ym);

	du_r[IND(x, y)] = (1.f - omega) * l_du[ly + BY][lx + BX] +
					  omega * (-J13 - J12 * l_dv[ly + BY][lx + BX] +

					  yp * (l_u[ly + BY + 1][lx + BX] - l_u[ly + BY][lx + BX]) + ym * (l_u[ly + BY - 1][lx + BX] - l_u[ly + BY][lx + BX]) +
			    	  xp * (l_u[ly + BY][lx + BX + 1] - l_u[ly + BY][lx + BX]) + xm * (l_u[ly + BY][lx + BX - 1] - l_u[ly + BY][lx + BX]) +

					  yp * l_du[ly + BY + 1][lx + BX] + ym * l_du[ly + BY - 1][lx + BX] +
					  xp * l_du[ly + BY][lx + BX + 1] + xm * l_du[ly + BY][lx + BX - 1]) / (J11 + sum);

	dv_r[IND(x, y)] = (1.f - omega) * l_dv[ly + BY][lx + BX] +
					  omega * (-J23 - J12 * l_du[ly + BY][lx + BX] +

					  yp * (l_v[ly + BY + 1][lx + BX] - l_v[ly + BY][lx + BX]) + ym * (l_v[ly + BY - 1][lx + BX] - l_v[ly + BY][lx + BX]) +
					  xp * (l_v[ly + BY][lx + BX + 1] - l_v[ly + BY][lx + BX]) + xm * (l_v[ly + BY][lx + BX - 1] - l_v[ly + BY][lx + BX]) +

					  yp * l_dv[ly + BY + 1][lx + BX] + ym * l_dv[ly + BY - 1][lx + BX] +
					  xp * l_dv[ly + BY][lx + BX + 1] + xm * l_dv[ly + BY][lx + BX - 1]) / (J22 + sum);
}

__kernel void Zero(
	__global			float*  d_mem			//  0 out	 : device memory filled with zeros
	)
{
	d_mem[get_global_id(0)] = 0.f;
}

