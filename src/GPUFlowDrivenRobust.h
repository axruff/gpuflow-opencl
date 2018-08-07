#pragma once

#include "OpticalFlowBase.h"
#include "Common.h"

class GPUFlowDrivenRobust :
	public OpticalFlowBase
{
private:
	cl_context m_clContext;
	cl_command_queue m_clCommandQueue;
	size_t m_localWorkSize[2];

	cl_program m_clProgram;
	cl_kernel m_clSolverKernel;
	cl_kernel m_clComputePhiKsiKernel;

	cl_mem m_d_Img_1;
	cl_mem m_d_Img_2;
	cl_mem m_d_du;
	cl_mem m_d_dv;
	cl_mem m_d_du_r;
	cl_mem m_d_dv_r;
	cl_mem m_d_u;
	cl_mem m_d_v;
	cl_mem m_d_phi;
	cl_mem m_d_ksi;

	int m_data_size;
	int m_inner_iterations;
	float m_e_smooth;
	float m_e_data;
public:
	GPUFlowDrivenRobust(const Image& img1, const Image& img2, int warp_levels, float warp_scale, int solver_iterations, int inner_iterations, float alpha, float omega, float e_smooth, float e_data,
		cl_context clContext, cl_command_queue clCommandQueue, int localWorkSize[2]);
	~GPUFlowDrivenRobust();
	void computeFlow(Image& u, Image& v);
	bool initResources(cl_context context, cl_device_id device);
	void releaseResources();
private:
	void solveDifference(Image& img_1, Image& img_2, Image& du, Image& dv, Image& u, Image& v, float hx, float hy);
};

