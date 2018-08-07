#pragma once

#include "OpticalFlowBase.h"

class CPUOpticalFlow :
	public OpticalFlowBase
{
public:
	CPUOpticalFlow(const Image& img1, const Image& img2, int warp_levels, float warp_scale, int solver_iterations, float alpha, float omega);
	~CPUOpticalFlow();

	void computeFlow(Image& u, Image& v);
private:
	void solveDifference(Image& img_1, Image& img_2, Image& du, Image& dv, const Image& u, const Image& v, float hx, float hy, float alpha, float omega);
};

