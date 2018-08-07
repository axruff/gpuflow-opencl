#include "CPUOpticalFlow.h"

#include <algorithm>
#include <iostream>
#include <cmath>

CPUOpticalFlow::CPUOpticalFlow(const Image& img1, const Image& img2, int warp_levels, float warp_scale, int solver_iterations, float alpha, float omega)
	: OpticalFlowBase(img1, img2, warp_levels, warp_scale, solver_iterations, alpha, omega)
{
}

CPUOpticalFlow::~CPUOpticalFlow()
{
}

void CPUOpticalFlow::computeFlow(Image& u, Image& v)
{
	int level_width;	// size in x - direction(current resolution)
	int level_height;	// size in x-direction (current resolution)
	float hx;			// spacing in x-direction (current resol.) 
	float hy;			// spacing in y-direction (current resol.) 

	Image img_1_res(m_source_img_1.width(), m_source_img_1.height(), 1, 1);	// 1st resampled image
	Image img_2_res(m_source_img_1.width(), m_source_img_1.height(), 1, 1); // 2nd resampled image
	Image img_2_br(m_source_img_1.width(), m_source_img_1.height(), 1, 1);  // 2nd warped image

	Image du(m_source_img_1.width(), m_source_img_1.height(), 1, 1);	// x-component of flow increment
	Image dv(m_source_img_1.width(), m_source_img_1.height(), 1, 1);	// y-component of flow increment

	int current_warp_level = std::min(m_warp_levels, computeMaxWarpLevels()) - 1;
	
	// initialize output flow arrays
	u.reinit(m_source_img_1.width(), m_source_img_1.height(), 1, 1, 1, 1);
	v.reinit(m_source_img_1.width(), m_source_img_1.height(), 1, 1, 1, 1);

	while (current_warp_level >= 0) {
		// compute level sizes
		level_width = static_cast<int>(ceil(m_source_img_1.width() * pow(m_warp_scale, current_warp_level)));
		level_height = static_cast<int>(ceil(m_source_img_1.height() * pow(m_warp_scale, current_warp_level)));
		hx = m_source_img_1.width() / static_cast<float>(level_width);
		hy = m_source_img_1.height() / static_cast<float>(level_height);

		std::cout << "Solve level: " << current_warp_level << " (" << level_width << "x" << level_height << ")" << std::endl;

		// perform resampling of images
		if (current_warp_level == 0) {
			img_1_res = m_source_img_1;
			img_2_res = m_source_img_2;
		} else {
			Image::resampleAreaBasedWithoutReallocating(m_source_img_1, img_1_res, level_width, level_height);
			Image::resampleAreaBasedWithoutReallocating(m_source_img_2, img_2_res, level_width, level_height);
		}
		// perform resampling of displacement field
		Image::resampleAreaBasedWithoutReallocating(u, du, level_width, level_height);
		Image::resampleAreaBasedWithoutReallocating(v, dv, level_width, level_height);
		u = du;
		v = dv;

		// perform backward registration
		Image::backwardRegistration(img_1_res, img_2_res, img_2_br, u, v, hx, hy);

		// solve difference problem at current resolution to obtain increment
		solveDifference(img_1_res, img_2_br, du, dv, u, v, hx, hy, m_alpha, m_omega);

		// add solved increment to the global flow
		u += du;
		v += dv;

		// go to the next level
		current_warp_level--;
	}
}

void CPUOpticalFlow::solveDifference(	   Image& img_1,	// in     : 1st image                                
										   Image& img_2,	// in     : 2nd image
										   Image& du,		// in+out : x-component of flow increment
										   Image& dv,		// in+out : y-component of flow increment
									 const Image& u,		// in	  : x-component of flow field
									 const Image& v,		// in	  : y-component of flow field
										   float hx,		// in     : grid spacing in x-direction
										   float hy,		// in     : grid spacing in y-direction
										   float alpha,		// in     : smoothness weight
										   float omega)		// in     : SOR overrelaxation parameter
{
	float hx_2, hy_2;		// time saver variables                              
	float xp, xm, yp, ym;	// neighbourhood weights                             
	float sum;              // central weight      

	int width = img_1.actual_width();
	int height = img_1.actual_height();
	
	hx_2 = alpha / (hx * hx);
	hy_2 = alpha / (hy * hy);
	
	du.setActualSize(width, height);
	dv.setActualSize(width, height);
	du.zeroData();
	dv.zeroData();

	img_1.fillBoudaries();
	img_2.fillBoudaries();

	// Compute motion tensor
	float* J11 = new float[width * height];
	float* J22 = new float[width * height];
	float* J12 = new float[width * height];
	float* J13 = new float[width * height];
	float* J23 = new float[width * height];

#define JIND(X, Y) ((Y) * width + (X))

	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			// Derivatives variables
			float fx = (img_1.pixel_r(x + 1, y) - img_1.pixel_r(x - 1, y) + img_2.pixel_r(x + 1, y) - img_2.pixel_r(x - 1, y)) / (4.f * hx);
			float fy = (img_1.pixel_r(x, y + 1) - img_1.pixel_r(x, y - 1) + img_2.pixel_r(x, y + 1) - img_2.pixel_r(x, y - 1)) / (4.f * hy);
			float ft = img_2.pixel_r(x, y) - img_1.pixel_r(x, y);
			J11[JIND(x, y)] = fx*fx;
			J22[JIND(x, y)] = fy*fy;
			J12[JIND(x, y)] = fx*fy;
			J13[JIND(x, y)] = fx*ft;
			J23[JIND(x, y)] = fy*ft;

		}
	}

	Image du_r;
	Image dv_r;

	// double buffering
	du_r.reinit(du.width(), du.height(), du.actual_width(), du.actual_height(), 1, 1);
	dv_r.reinit(du.width(), du.height(), du.actual_width(), du.actual_height(), 1, 1);
	  
	// For all iterations		      
	for (int k = 0; k < m_solver_iterations; k++) {
		// For all image pixels
		for (int y = 0; y < height; y++) {
			for (int x = 0; x < width; x++) {
				// Compute weights 
				xp = (x < width - 1)	* hx_2;
				xm = (x > 0)			* hx_2;
				yp = (y < height - 1)	* hy_2;
				ym = (y > 0)			* hy_2;
				
				sum = (xp + xm + yp + ym);
				
				du_r.pixel_w(x, y) = (1.f - omega) * du.pixel_r(x, y) +
								   omega * ( -J13[JIND(x, y)] - J12[JIND(x, y)] * dv.pixel_r(x, y) +

								   yp * (u.pixel_r(x, y + 1) - u.pixel_r(x, y)) + ym * (u.pixel_r(x, y -1) - u.pixel_r(x, y)) + 
								   xp * (u.pixel_r(x + 1, y) - u.pixel_r(x, y)) + xm * (u.pixel_r(x - 1, y)- u.pixel_r(x, y)) +

								   yp * du.pixel_r(x, y + 1) + ym * du.pixel_r(x, y - 1) + 
								   xp * du.pixel_r(x + 1, y) + xm * du.pixel_r(x - 1, y)) / (J11[JIND(x, y)] + sum);

				dv_r.pixel_w(x, y) = (1.f - omega) * dv.pixel_r(x, y) +
								   omega * ( -J23[JIND(x, y)] - J12[JIND(x, y)] * du.pixel_r(x, y) +

								   yp * (v.pixel_r(x, y + 1) - v.pixel_r(x, y)) + ym * (v.pixel_r(x, y - 1) - v.pixel_r(x, y)) + 
								   xp * (v.pixel_r(x + 1, y) - v.pixel_r(x, y)) + xm * (v.pixel_r(x - 1, y) - v.pixel_r(x, y)) +

								   yp * dv.pixel_r(x, y + 1) + ym * dv.pixel_r(x, y - 1) + 
								   xp * dv.pixel_r(x + 1, y) + xm * dv.pixel_r(x - 1, y) ) / (J22[JIND(x, y)] + sum);
			}
		}
		du.swap_data(du_r);
		dv.swap_data(dv_r);
	}

	delete[] J11;
	delete[] J22;
	delete[] J12;
	delete[] J13;
	delete[] J23;
}