#define IND(X, Y) (((Y) + by) * pitch + ((X) + bx))

__kernel void Solver(
	__global	const	float*	d_img_1,	//  0 in     : 1st image 
	__global	const	float*	d_img_2,	//  1 in     : 2nd image
	__global	const	float*  du,			//  2 in	 : x-component of flow increment
	__global	const	float*  dv,			//  3 in	 : y-component of flow increment
	__global	const	float*  u,			//  4 in	 : x-component of flow field
	__global	const	float*  v,			//  5 in	 : y-component of flow field
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

	dv_r[IND(x, y)] = (1.f - omega) * dv[IND(x, y)] +
		omega * (-J23 - J12 * du[IND(x, y)] +

		yp * (v[IND(x, y + 1)] - v[IND(x, y)]) + ym * (v[IND(x, y - 1)] - v[IND(x, y)]) +
		xp * (v[IND(x + 1, y)] - v[IND(x, y)]) + xm * (v[IND(x - 1, y)] - v[IND(x, y)]) +

		yp * dv[IND(x, y + 1)] + ym * dv[IND(x, y - 1)] +
		xp * dv[IND(x + 1, y)] + xm * dv[IND(x - 1, y)]) / (J22 + sum);
}

__kernel void Zero(
	__global			float*  d_mem		//  0 out	 : device memory filled with zeros
	)
{
	d_mem[get_global_id(0)] = 0.f;
}

__kernel void Add(
	__global			float4*  d_dst,		//  0 in:out : sum
	__global	const	float4*  d_src		//  1 in	 : add
	)
{
	d_dst[get_global_id(0)] += d_src[get_global_id(0)];
}

__kernel void BackwardRegistration(
	__global	const	float*  d_img_1,	//  0 in	 : 1st image
	__global	const	float*  d_img_2,	//  1 in	 : 2nd image
	__global	const	float*  u,			//  2 in	 : x-component of flow field
	__global	const	float*  v,			//  3 in	 : y-component of flow field
						float	hx,			//  4 in     : grid spacing in x-direction
						float	hy,			//  5 in     : grid spacing in y-direction
						int		bx,			//  6 in	 : x-border size
						int		by,         //  7 in     : y-border size
						int		width,		//  8 in     : image width
						int		height,		//  9 in     : image height
						int		pitch,		// 10 in     : image pitch	
	__global			float*  d_img_2_br	// 11 out	 : 2nd image (motion compensated)
	)
{
	int x = get_global_id(0);
	int y = get_global_id(1);

	if (x >= width || y >= height) {
		return;
	}

	int   yy, xx;               // pixel coordinates                              
	float yy_fp, xx_fp;         // subpixel coordinates                           
	float delta_y, delta_x;     // subpixel displacement                          

	float hx_1 = 1.f / hx;
	float hy_1 = 1.f / hy;

	// Compute subpixel location 
	yy_fp = y + v[IND(x, y)] * hy_1;
	xx_fp = x + u[IND(x, y)] * hx_1;
	
	// If the required image information is out of bounds 
	if ((yy_fp < 0) || (xx_fp < 0) || (yy_fp > (height - 1)) || (xx_fp > (width - 1))){
		// assume zero flow, i.e. set warped 2nd image to 1st image 
		d_img_2_br[IND(x, y)] = d_img_1[IND(x, y)];
	} else {
		// compute integer index of upper left pixel 
		yy = floor(yy_fp);
		xx = floor(xx_fp);

		// compute subpixel displacement 
		delta_y = yy_fp - yy;
		delta_x = xx_fp - xx;

		// perform bilinear interpolation 
		d_img_2_br[IND(x, y)] = (1.f - delta_y) * (1.f - delta_x)	* d_img_2[IND(xx, yy)]
							  + (1.f - delta_y) * delta_x			* d_img_2[IND(xx + 1, yy)]
							  + delta_y		    * (1.f - delta_x)	* d_img_2[IND(xx, yy + 1)]
							  + delta_y		    * delta_x			* d_img_2[IND(xx + 1, yy + 1)];
	}
}

__kernel void ReflectHorizontalBoudaries(
	__global			float*  d_img,		//  0 in	 : image
						int		bx,			//  1 in	 : x-border size
						int		by,         //  2 in     : y-border size
						int		width,		//  3 in     : image width
						int		height,		//  4 in     : image height
						int		pitch		//  5 in     : image pitch	
	)
{
	size_t x = get_global_id(0);
	if (x >= width) {
		return;
	}
	d_img[IND(x, -by)]	           = d_img[IND(x, by)];
	d_img[IND(x, height - 1 + by)] = d_img[IND(x, height - 1 - by)];
}

__kernel void ReflectVerticalBoudaries(
	__global			float*  d_img,		//  0 in	 : image
						int		bx,			//  1 in	 : x-border size
						int		by,         //  2 in     : y-border size
						int		width,		//  3 in     : image width
						int		height,		//  4 in     : image height
						int		pitch		//  5 in     : image pitch	
	)
{
	size_t y = get_global_id(0);
	if (y >= height) {
		return;
	}
	d_img[IND(-bx, y)]			   = d_img[IND(bx, y)];
	d_img[IND(width  - 1 + bx, y)] = d_img[IND(width - 1 - by, y)];
}

__kernel void ResampleY(
	__global	const	float*  d_src,		//  0 in	 : source image
	__global			float*  d_dst,		//  1 out	 : resampled image
						int		width,		//  2 in     : image width
						int		src_height,	//  3 in     : image height
						int		dst_height,	//  4 in     : image height
						int		pitch		//  5 in     : image pitch	
	)
{
	const int bx = 1;
	const int by = 1;

	size_t x = get_global_id(0);

	if (x >= width) {
		return;
	}

	int    sy;				/* loop variables		*/
	float  hs, hd;          /* grid sizes           */
	float  sleft, sright;   /* boundaries           */
	float  dleft, dright;   /* boundaries           */
	float  fac;             /* normalization factor */

	hs = 1.0f / (float)src_height;     /* grid size of src               */
	hd = 1.0f / (float)dst_height;     /* grid size of dst               */
	sleft = 0.0f;					   /* left interval boundary of src  */
	dleft = 0.0f;                      /* left interval boundary of dst  */
	sy = 0;							   /* index for src                  */
	fac = hs / hd;                     /* for normalization              */

	__local float pixel;

	for (int y = 0; y < dst_height; y++) {

		/* calculate right interval boundaries */
		sright = sleft + hs;
		dright = dleft + hd;

		if (sright > dright)  {
			/* since sleft <= dleft, the entire d-cell i is in the s-cell k */
			pixel = d_src[IND(x, sy)];
		} else {
			/* consider fraction alpha of the s-cell k in d-cell i */
			pixel = (sright - dleft) * src_height * d_src[IND(x, sy++)];

			/* update */
			sright = sright + hs;

			/* consider entire u-cells inside v-cell i */
			while (sright <= dright)
			/* s-cell sy lies entirely in v-cell y; sum up */
			{
				pixel += d_src[IND(x, sy)];
				sright = sright + hs;
				sy = min(++sy, src_height - 1);
			}
			/* consider fraction beta of the u-cell k in v-cell i */
			pixel += (1.f - (sright - dright) * src_height) * d_src[IND(x, sy)];

			/* normalization */
			pixel *= fac;
		}
		/* update now it holds: sleft <= dleft */
		sleft = sright - hs;
		dleft = dright;
 
		/* write data back from local memory to global */
		d_dst[IND(x, y)] = pixel;
	}
}

__kernel void ResampleX(
	__global	const	float*  d_src,		//  0 in	 : source image
	__global			float*  d_dst,		//  1 out	 : resampled image
						int		height,		//  2 in     : image height
						int		src_width,	//  3 in     : image width
						int		dst_width,	//  4 in     : image width
						int		pitch		//  5 in     : image pitch	
	)
{
	const int bx = 1;
	const int by = 1;

	size_t y = get_global_id(0);

	if (y >= height) {
		return;
	}

	int    sx;				/* loop variables		*/
	float  hs, hd;          /* grid sizes           */
	float  sleft, sright;   /* boundaries           */
	float  dleft, dright;   /* boundaries           */
	float  fac;             /* normalization factor */

	hs = 1.0f / (float)src_width;      /* grid size of src               */
	hd = 1.0f / (float)dst_width;      /* grid size of dst               */
	sleft = 0.0f;					   /* left interval boundary of src  */
	dleft = 0.0f;                      /* left interval boundary of dst  */
	sx = 0;							   /* index for src                  */
	fac = hs / hd;                     /* for normalization              */

	__local float pixel;

	for (int x = 0; x < dst_width; x++) {

		/* calculate right interval boundaries */
		sright = sleft + hs;
		dright = dleft + hd;

		if (sright > dright)  {
			/* since sleft <= dleft, the entire d-cell i is in the s-cell k */
			pixel = d_src[IND(sx, y)];
		} else {
			/* consider fraction alpha of the s-cell k in d-cell i */
			pixel = (sright - dleft) * src_width * d_src[IND(sx++, y)];

			/* update */
			sright = sright + hs;

			/* consider entire u-cells inside v-cell i */
			while (sright <= dright)
				/* s-cell sy lies entirely in v-cell y; sum up */
			{
				pixel += d_src[IND(sx, y)];
				sright = sright + hs;
				sx = min(++sx, src_width - 1);
			}
			/* consider fraction beta of the u-cell k in v-cell i */
			pixel += (1.f - (sright - dright) * src_width) * d_src[IND(sx, y)];

			/* normalization */
			pixel *= fac;
		}
		/* update now it holds: sleft <= dleft */
		sleft = sright - hs;
		dleft = dright;

		/* write data back from local memory to global */
		d_dst[IND(x, y)] = pixel;
	}

}