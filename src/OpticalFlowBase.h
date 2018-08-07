#pragma once

#include "Image.h"

class OpticalFlowBase
{
protected:
	const Image&	m_source_img_1;
	const Image&	m_source_img_2;

	int		m_warp_levels;
	float	m_warp_scale;
	int		m_solver_iterations;
	float	m_alpha;
	float	m_omega;

public:
	OpticalFlowBase(const Image& img1, const Image& img2, int warp_levels, float warp_scale, int solver_iterations, float alpha, float omega);

	virtual void computeFlow(Image& u, Image& v) = 0;

protected:
	int computeMaxWarpLevels() const;

};

