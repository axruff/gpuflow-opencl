#include "OpticalFlowBase.h"

// Linux declaration
#ifndef _WIN32 
	#include <cmath>
	using namespace std;
#endif

OpticalFlowBase::OpticalFlowBase(const Image& img1, const Image& img2, int warp_levels, float warp_scale, int solver_iterations, float alpha, float omega)
	: m_source_img_1(img1), m_source_img_2(img2), m_warp_levels(warp_levels), m_warp_scale(warp_scale), m_solver_iterations(solver_iterations),
	m_alpha(alpha), m_omega(omega)
{	
}

int OpticalFlowBase::computeMaxWarpLevels() const
// compute maximum number of warping levels for given image size and warping 
// reduction factor 
{
	int nx_orig = m_source_img_1.width();
	int ny_orig = m_source_img_1.height();

	int   i;               // level counter                                 
	int nx, ny;           // reduced dimensions                            
	int nx_old, ny_old;   // reduced dimensions                            

	nx_old = nx_orig;
	ny_old = ny_orig;

	for (i = 1;; i++)
	{
		nx = (int) ceil((float) nx_orig * pow(m_warp_scale, i));
		ny = (int) ceil((float) ny_orig * pow(m_warp_scale, i));

		if ((nx<4) || (ny<4)) break;

		nx_old = nx;
		ny_old = ny;
	}
	if ((nx == 1) || (ny == 1)) i--;

	return i;
}
