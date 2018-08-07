#pragma once

#include "OpticalFlowBase.h"
#include "Common.h"

class GPUFullOpticalFlow :
	public OpticalFlowBase
{
private:
	cl_context m_clContext;
	cl_command_queue m_clCommandQueue;
	size_t m_localWorkSize[2];

	cl_program m_clProgram;
	cl_kernel m_clSolverKernel;
	cl_kernel m_clZeroKernel;
	cl_kernel m_clAddKernel;
	cl_kernel m_clBackwardRegistrationKernel;
	cl_kernel m_clReflectHorizontalBoudariesKernel;
	cl_kernel m_clReflectVerticalBoudariesKernel;
	cl_kernel m_clResampleXKernel;
	cl_kernel m_clResampleYKernel;

	cl_mem m_d_src_Img1;
	cl_mem m_d_src_Img2;
	cl_mem m_d_Img_1;
	cl_mem m_d_Img_2;
	cl_mem m_d_Img_2_br;
	cl_mem m_d_du;
	cl_mem m_d_dv;
	cl_mem m_d_du_r;
	cl_mem m_d_dv_r;
	cl_mem m_d_u;
	cl_mem m_d_v;

	int m_data_size;
public:
	GPUFullOpticalFlow(const Image& img1, const Image& img2, int warp_levels, float warp_scale, int solver_iterations, float alpha, float omega,
		cl_context clContext, cl_command_queue clCommandQueue, int localWorkSize[2]);
	~GPUFullOpticalFlow();

	void computeFlow(Image& u, Image& v);
	bool initResources(cl_context context, cl_device_id device);
	void releaseResources();
private:
	void solveDifference(float hx, float hy, int width, int height);
	void backwardRegistration(float hx, float hy, int width, int height);
	void reflectBoudaries(int width, int height);
	void resampleAreaBased(cl_mem src, cl_mem dst, int src_width, int src_height,  int dst_width, int dst_height);
	void resample_y(cl_mem src, cl_mem dst, int src_width, int src_height, int dst_width, int dst_height);
	void resample_x(cl_mem src, cl_mem dst, int src_width, int src_height, int dst_width, int dst_height);
	void addFlowIncrement();
	void zeroDeviceBuffer(cl_mem mem);
};

