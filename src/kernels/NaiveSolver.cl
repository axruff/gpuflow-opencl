
#define IND(X, Y) (((Y) + by) * pitch + ((X) + bx))

__kernel void NaiveSolver(
	__global	const	float*	d_img_1,	//  0 in     : 1st image 
	__global	const	float*	d_img_2,	//  1 in     : 2nd image
	__global			float*  du,			//  2 in	 : x-component of flow increment
	__global			float*  dv,			//  3 in	 : y-component of flow increment
	__global			float*  u,			//  4 in	 : x-component of flow field
	__global			float*  v,			//  5 in	 : y-component of flow field
						float	hx,			//  6 in     : grid spacing in x-direction
						float	hy,			//  7 in     : grid spacing in y-direction
						float	alpha,		//  8 in     : smoothness weight
						float	omega,		//  9 in     : SOR overrelaxation parameter
						int		bx,			// 10 in	 : x-border size
						int		by,         // 11 in     : y-border size
						int		width,		// 12 in     : image width
						int		height,		// 13 in     : image height
						int		pitch,		// 14 in     : image pitch
	__global			float*	du_r,		// 15 out	 : du result
	__global			float*	dv_r		// 16 out	 : dv result
)
{
	size_t x = get_global_id(0);
	size_t y = get_global_id(1);

	if (x >= width || y >= height) {
		return;
	}

	float hx_2 = alpha / (hx * hx);
	float hy_2 = alpha / (hy * hy);
	
	// Derivatives variables
	float fx = (d_img_1[IND(x + 1, y)] - d_img_1[IND(x - 1, y)] + d_img_2[IND(x + 1, y)] - d_img_2[IND(x - 1, y)]) / (4.f * hx);
	float fy = (d_img_1[IND(x, y + 1)] - d_img_1[IND(x, y - 1)] + d_img_2[IND(x, y + 1)] - d_img_2[IND(x, y - 1)]) / (4.f * hy);
	float ft = d_img_2[IND(x, y)] - d_img_1[IND(x, y)];
	
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

	du_r[IND(x, y)] = (1.f - omega) * du[IND(x, y)] +
					omega * (-J13 - J12 * dv[IND(x, y)] +

					yp * (u[IND(x, y + 1)] - u[IND(x, y)]) + ym * (u[IND(x, y - 1)] - u[IND(x, y)]) +
					xp * (u[IND(x + 1, y)] - u[IND(x, y)]) + xm * (u[IND(x - 1, y)] - u[IND(x, y)]) +

					yp * du[IND(x, y + 1)] + ym * du[IND(x, y - 1)] +
					xp * du[IND(x + 1, y)] + xm * du[IND(x - 1, y)]) / (J11 + sum);

	dv_r[IND(x, y)] = (1.f - omega) * dv[IND(x, y)]+
					omega * (-J23 - J12 * du[IND(x, y)]+

					yp * (v[IND(x, y + 1)] - v[IND(x, y)]) + ym * (v[IND(x, y - 1)] - v[IND(x, y)]) +
					xp * (v[IND(x + 1, y)] - v[IND(x, y)]) + xm * (v[IND(x - 1, y)] - v[IND(x, y)]) +

					yp * dv[IND(x, y + 1)] + ym * dv[IND(x, y - 1)] +
					xp * dv[IND(x + 1, y)] + xm * dv[IND(x - 1, y)]) / (J22 + sum);
 }
