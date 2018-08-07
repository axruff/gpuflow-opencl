
#define IND(X, Y) (((Y) + by) * pitch + ((X) + bx))


__kernel void ComputePhiKsi(
	__global	const	float*	u,			//  0 in     : x-component of flow field 
	__global	const	float*	v,			//  1 in     : y-component of flow field
	__global	const	float*	du,			//  2 in     : x-component of flow increment 
	__global	const	float*	dv,			//  3 in     : y-component of flow increment
	__global			float*	phi,		//  4 out    : phi 
	__global			float*	ksi,		//  5 out    : ksi 
						float	e_smooth,	//  6 in	 : e_smooth
						float	hx,			//  7 in     : grid spacing in x-direction
						float	hy,			//  8 in     : grid spacing in y-direction
						int		bx,			//  9 in	 : x-border size
						int		by,         // 10 in     : y-border size
						int		width,		// 11 in     : image width
						int		height,		// 12 in     : image height
						int		pitch,		// 13 in     : image pitch
	__global	const	float*	d_img_1,	// 14 in     : 1st image 
	__global	const	float*	d_img_2,	// 15 in     : 2nd image
						float	e_data		// 16 in     : e_data
	)
{
	size_t x = get_global_id(0);
	size_t y = get_global_id(1);

	if (x >= width || y >= height) {
		return;
	}

	float ux = (u[IND(x + 1, y)] - u[IND(x - 1, y)]) / (2.f * hx);
	float uy = (u[IND(x, y + 1)] - u[IND(x, y - 1)]) / (2.f * hy);
	float vx = (v[IND(x + 1, y)] - v[IND(x - 1, y)]) / (2.f * hx);
	float vy = (v[IND(x, y + 1)] - v[IND(x, y - 1)]) / (2.f * hy);

	float dux = (du[IND(x + 1, y)] - du[IND(x - 1, y)]) / (2.f * hx);
	float duy = (du[IND(x, y + 1)] - du[IND(x, y - 1)]) / (2.f * hy);
	float dvx = (dv[IND(x + 1, y)] - dv[IND(x - 1, y)]) / (2.f * hx);
	float dvy = (dv[IND(x, y + 1)] - dv[IND(x, y - 1)]) / (2.f * hy);

	dux = ux + dux;
	duy = uy + duy;
	dvx = vx + dvx;
	dvy = vy + dvy;

	phi[IND(x, y)] = 1.f / (2.f * sqrt(dux*dux + duy*duy + dvx*dvx + dvy*dvy + e_smooth * e_smooth));

	// Derivatives variables
	float fx = (d_img_1[IND(x + 1, y)] - d_img_1[IND(x - 1, y)] + d_img_2[IND(x + 1, y)] - d_img_2[IND(x - 1, y)]) / (4.f * hx);
	float fy = (d_img_1[IND(x, y + 1)] - d_img_1[IND(x, y - 1)] + d_img_2[IND(x, y + 1)] - d_img_2[IND(x, y - 1)]) / (4.f * hy);
	float ft = d_img_2[IND(x, y)] - d_img_1[IND(x, y)];

	float J11 = fx * fx;
	float J22 = fy * fy;
	float J33 = ft * ft;
	float J12 = fx * fy;
	float J13 = fx * ft;
	float J23 = fy * ft;
	
	// Weight for data term
	float du_ = du[IND(x, y)];
	float dv_ = dv[IND(x, y)];

	float s = (J11 * du_ + J12 * dv_ + J13) * du_ +
			  (J12 * du_ + J22 * dv_ + J23) * dv_ +
			  (J13 * du_ + J23 * dv_ + J33);

	s = (s > 0) * s;

	// Penalizer function for data term
	ksi[IND(x, y)] = 1.f / (2.f * sqrt(s + e_data * e_data));
}

__kernel void Solver(
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
	__global			float*	dv_r,		// 16 out	 : dv result
	__global	const	float*	phi,		// 17 in     : precomputed phi
	__global	const	float*	ksi			// 18 in     : precomputed ksi
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
	float J33 = ft * ft;
	float J12 = fx * fy;
	float J13 = fx * ft;
	float J23 = fy * ft;

	// Compute weights 
	float xp = (x < width - 1)	* hx_2;
	float xm = (x > 0)			* hx_2;
	float yp = (y < height - 1)	* hy_2;
	float ym = (y > 0)			* hy_2;

	float phiLower = (phi[IND(x, y + 1)] + phi[IND(x, y)]) / 2.0;
	float phiUpper = (phi[IND(x, y - 1)] + phi[IND(x, y)]) / 2.0;
	float phiLeft  = (phi[IND(x - 1, y)] + phi[IND(x, y)]) / 2.0;
	float phiRight = (phi[IND(x + 1, y)] + phi[IND(x, y)]) / 2.0;

	float sumH = (xp*phiRight + xm*phiLeft + yp*phiLower + ym*phiUpper);

	float sumU = phiLower*yp*(u[IND(x, y + 1)] + du[IND(x, y + 1)] - u[IND(x, y)]) + phiUpper*ym*(u[IND(x, y - 1)] + du[IND(x, y - 1)] - u[IND(x, y)]) + 
				 phiRight*xp*(u[IND(x + 1, y)] + du[IND(x + 1, y)] - u[IND(x, y)]) + phiLeft*xm*(u[IND(x - 1, y)] + du[IND(x - 1, y)] - u[IND(x, y)]);

	float sumV = phiLower*yp*(v[IND(x, y + 1)] + dv[IND(x, y + 1)] - v[IND(x, y)]) + phiUpper*ym*(v[IND(x, y - 1)] + dv[IND(x, y - 1)] - v[IND(x, y)]) +
				 phiRight*xp*(v[IND(x + 1, y)] + dv[IND(x + 1, y)] - v[IND(x, y)]) + phiLeft*xm*(v[IND(x - 1, y)] + dv[IND(x - 1, y)] - v[IND(x, y)]);

	du_r[IND(x, y)] = (1.0 - omega)*du[IND(x, y)] +
					  omega*(ksi[IND(x, y)] * (-J13 - J12 * dv[IND(x, y)]) + sumU) / (ksi[IND(x, y)] * J11 + sumH);

	dv_r[IND(x, y)] = (1.0 - omega)*dv[IND(x, y)] +
					  omega*(ksi[IND(x, y)] * (-J23 - J12 * du[IND(x, y)]) + sumV) / (ksi[IND(x, y)] * J22 + sumH);
 }


