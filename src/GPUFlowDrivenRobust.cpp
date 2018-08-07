#include "GPUFlowDrivenRobust.h"

#include "CTimer.h"
#include <algorithm>
#include <sstream>

GPUFlowDrivenRobust::GPUFlowDrivenRobust(const Image& img1, const Image& img2, int warp_levels, float warp_scale, int solver_iterations, int inner_iterations, float alpha, float omega, float e_smooth, float e_data, 
	cl_context clContext, cl_command_queue clCommandQueue, int localWorkSize[2])
	: OpticalFlowBase(img1, img2, warp_levels, warp_scale, solver_iterations, alpha, omega), m_inner_iterations(inner_iterations), m_e_smooth(e_smooth), m_e_data(e_data),
	m_clContext(clContext), m_clCommandQueue(clCommandQueue),
	m_clProgram(NULL), m_clSolverKernel(NULL), m_clComputePhiKsiKernel(NULL),
	m_d_Img_1(NULL), m_d_Img_2(NULL), m_d_du(NULL), m_d_dv(NULL), m_d_u(NULL), m_d_v(NULL), m_d_phi(NULL),
	m_data_size(0)
{
	m_localWorkSize[0] = localWorkSize[0];
	m_localWorkSize[1] = localWorkSize[1];
}

GPUFlowDrivenRobust::~GPUFlowDrivenRobust()
{
}

bool GPUFlowDrivenRobust::initResources(cl_context context, cl_device_id device)
{
	cl_int cl_error;
	char * program_code;
	size_t program_size;

	LoadProgram("./src/kernels/FlowDrivenSolver.cl", &program_code, &program_size);

	// create a program object
	m_clProgram = clCreateProgramWithSource(context, 1, (const char**)&program_code, &program_size, &cl_error);
	V_RETURN_FALSE_CL(cl_error, "Failed to create program from file.");

	// buid program
	cl_error = clBuildProgram(m_clProgram, 1, &device, NULL, NULL, NULL);
	if (cl_error != CL_SUCCESS)
	{
		PrintBuildLog(m_clProgram, device);
		return false;
	}

	// create kernels
	m_clSolverKernel = clCreateKernel(m_clProgram, "Solver", &cl_error);
	V_RETURN_FALSE_CL(cl_error, "Failed to create kernel.");

	m_clComputePhiKsiKernel = clCreateKernel(m_clProgram, "ComputePhiKsi", &cl_error);
	V_RETURN_FALSE_CL(cl_error, "Failed to create kernel.");


	// create device resources
	int bx = 1;
	int by = 1;
	// temporal image to get right image sizes
	Image img(m_source_img_1.width(), m_source_img_1.height(), bx, by);
	int height = img.height();
	int pitch = img.pitch();
	m_data_size = pitch * (height + 2 * by) * sizeof(cl_float);

	m_d_Img_1 = clCreateBuffer(context, CL_MEM_READ_ONLY, m_data_size, NULL, &cl_error);
	V_RETURN_FALSE_CL(cl_error, "Error allocating device memory");
	m_d_Img_2 = clCreateBuffer(context, CL_MEM_READ_ONLY, m_data_size, NULL, &cl_error);
	V_RETURN_FALSE_CL(cl_error, "Error allocating device memory");
	m_d_u = clCreateBuffer(context, CL_MEM_READ_ONLY, m_data_size, NULL, &cl_error);
	V_RETURN_FALSE_CL(cl_error, "Error allocating device memory");
	m_d_v = clCreateBuffer(context, CL_MEM_READ_ONLY, m_data_size, NULL, &cl_error);
	V_RETURN_FALSE_CL(cl_error, "Error allocating device memory");
	m_d_du = clCreateBuffer(context, CL_MEM_READ_WRITE, m_data_size, NULL, &cl_error);
	V_RETURN_FALSE_CL(cl_error, "Error allocating device memory");
	m_d_dv = clCreateBuffer(context, CL_MEM_READ_WRITE, m_data_size, NULL, &cl_error);
	V_RETURN_FALSE_CL(cl_error, "Error allocating device memory");
	m_d_du_r = clCreateBuffer(context, CL_MEM_READ_WRITE, m_data_size, NULL, &cl_error);
	V_RETURN_FALSE_CL(cl_error, "Error allocating device memory");
	m_d_dv_r = clCreateBuffer(context, CL_MEM_READ_WRITE, m_data_size, NULL, &cl_error);
	V_RETURN_FALSE_CL(cl_error, "Error allocating device memory");
	m_d_phi = clCreateBuffer(context, CL_MEM_READ_WRITE, m_data_size, NULL, &cl_error);
	V_RETURN_FALSE_CL(cl_error, "Error allocating device memory");
	m_d_ksi = clCreateBuffer(context, CL_MEM_READ_WRITE, m_data_size, NULL, &cl_error);
	V_RETURN_FALSE_CL(cl_error, "Error allocating device memory");

	// bind kernel arguments (constant for all iterations)
	cl_error = clSetKernelArg(m_clSolverKernel, 0, sizeof(cl_mem), (void*)&m_d_Img_1);
	cl_error |= clSetKernelArg(m_clSolverKernel, 1, sizeof(cl_mem), (void*)&m_d_Img_2);

	cl_error |= clSetKernelArg(m_clSolverKernel, 4, sizeof(cl_mem), (void*)&m_d_u);
	cl_error |= clSetKernelArg(m_clSolverKernel, 5, sizeof(cl_mem), (void*)&m_d_v);

	cl_error |= clSetKernelArg(m_clSolverKernel, 8, sizeof(cl_float), (void*)&m_alpha);
	cl_error |= clSetKernelArg(m_clSolverKernel, 9, sizeof(cl_float), (void*)&m_omega);
	cl_error |= clSetKernelArg(m_clSolverKernel, 10, sizeof(cl_int), (void*)&bx);
	cl_error |= clSetKernelArg(m_clSolverKernel, 11, sizeof(cl_int), (void*)&by);

	cl_error |= clSetKernelArg(m_clSolverKernel, 14, sizeof(cl_int), (void*)&pitch);

	cl_error |= clSetKernelArg(m_clSolverKernel, 17, sizeof(cl_mem), (void*)&m_d_phi);
	cl_error |= clSetKernelArg(m_clSolverKernel, 18, sizeof(cl_mem), (void*)&m_d_ksi);
	V_RETURN_FALSE_CL(cl_error, "Error setting kernel arguments");

	cl_error  = clSetKernelArg(m_clComputePhiKsiKernel, 0, sizeof(cl_mem), (void*)&m_d_u);
	cl_error |= clSetKernelArg(m_clComputePhiKsiKernel, 1, sizeof(cl_mem), (void*)&m_d_v);
	
	cl_error |= clSetKernelArg(m_clComputePhiKsiKernel, 4, sizeof(cl_mem), (void*)&m_d_phi);
	cl_error |= clSetKernelArg(m_clComputePhiKsiKernel, 5, sizeof(cl_mem), (void*)&m_d_ksi);
	cl_error |= clSetKernelArg(m_clComputePhiKsiKernel, 6, sizeof(cl_float), (void*)&m_e_smooth);

	cl_error |= clSetKernelArg(m_clComputePhiKsiKernel, 9, sizeof(cl_int), (void*)&bx);
	cl_error |= clSetKernelArg(m_clComputePhiKsiKernel, 10, sizeof(cl_int), (void*)&by);

	cl_error |= clSetKernelArg(m_clComputePhiKsiKernel, 13, sizeof(cl_int), (void*)&pitch);
	cl_error |= clSetKernelArg(m_clComputePhiKsiKernel, 14, sizeof(cl_mem), (void*)&m_d_Img_1);
	cl_error |= clSetKernelArg(m_clComputePhiKsiKernel, 15, sizeof(cl_mem), (void*)&m_d_Img_2);
	cl_error |= clSetKernelArg(m_clComputePhiKsiKernel, 16, sizeof(cl_float), (void*)&m_e_data);
	V_RETURN_FALSE_CL(cl_error, "Error setting kernel arguments");

	return true;
}

void GPUFlowDrivenRobust::releaseResources()
{
	SAFE_RELEASE_MEMOBJECT(m_d_Img_1);
	SAFE_RELEASE_MEMOBJECT(m_d_Img_2);
	SAFE_RELEASE_MEMOBJECT(m_d_du);
	SAFE_RELEASE_MEMOBJECT(m_d_dv);
	SAFE_RELEASE_MEMOBJECT(m_d_du_r);
	SAFE_RELEASE_MEMOBJECT(m_d_dv_r);
	SAFE_RELEASE_MEMOBJECT(m_d_u);
	SAFE_RELEASE_MEMOBJECT(m_d_v);
	SAFE_RELEASE_MEMOBJECT(m_d_phi);
	SAFE_RELEASE_MEMOBJECT(m_d_ksi);

	SAFE_RELEASE_KERNEL(m_clSolverKernel);
	SAFE_RELEASE_KERNEL(m_clComputePhiKsiKernel);
	SAFE_RELEASE_PROGRAM(m_clProgram);
}

void GPUFlowDrivenRobust::computeFlow(Image& u, Image& v)
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

	int current_warp_level = min(m_warp_levels, computeMaxWarpLevels()) - 1;

	// initialize output flow arrays
	u.reinit(m_source_img_1.width(), m_source_img_1.height(), 1, 1, 1, 1);
	v.reinit(m_source_img_1.width(), m_source_img_1.height(), 1, 1, 1, 1);

	while (current_warp_level >= 0) {
		// compute level sizes
		level_width = static_cast<int>(ceil(m_source_img_1.width() * pow(m_warp_scale, current_warp_level)));
		level_height = static_cast<int>(ceil(m_source_img_1.height() * pow(m_warp_scale, current_warp_level)));
		hx = m_source_img_1.width() / static_cast<float>(level_width);
		hy = m_source_img_1.height() / static_cast<float>(level_height);

		std::cout << "Solve level: " << current_warp_level << " (" << level_width << "x" << level_height << ") \t "; // << std::endl;

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
		solveDifference(img_1_res, img_2_br, du, dv, u, v, hx, hy);

		// add solved increment to the global flow
		u += du;
		v += dv;

		// go to the next level
		current_warp_level--;
	}
}

void GPUFlowDrivenRobust::solveDifference(Image& img_1, Image& img_2, Image& du, Image& dv, Image& u, Image& v, float hx, float hy)
{
	int width = img_1.actual_width();
	int height = img_1.actual_height();

	img_1.fillBoudaries();
	img_2.fillBoudaries();

	du.setActualSize(width, height);
	dv.setActualSize(width, height);
	du.zeroData();
	dv.zeroData();

	// copy data to device
	V_RETURN_CL(clEnqueueWriteBuffer(m_clCommandQueue, m_d_Img_1, CL_FALSE, 0, m_data_size, img_1.data_ptr(), 0, NULL, NULL), "Error copying input data to device!");
	V_RETURN_CL(clEnqueueWriteBuffer(m_clCommandQueue, m_d_Img_2, CL_FALSE, 0, m_data_size, img_2.data_ptr(), 0, NULL, NULL), "Error copying input data to device!");
	V_RETURN_CL(clEnqueueWriteBuffer(m_clCommandQueue, m_d_du, CL_FALSE, 0, m_data_size, du.data_ptr(), 0, NULL, NULL), "Error copying input data to device!");
	V_RETURN_CL(clEnqueueWriteBuffer(m_clCommandQueue, m_d_dv, CL_FALSE, 0, m_data_size, dv.data_ptr(), 0, NULL, NULL), "Error copying input data to device!");
	V_RETURN_CL(clEnqueueWriteBuffer(m_clCommandQueue, m_d_du_r, CL_FALSE, 0, m_data_size, du.data_ptr(), 0, NULL, NULL), "Error copying input data to device!");
	V_RETURN_CL(clEnqueueWriteBuffer(m_clCommandQueue, m_d_dv_r, CL_FALSE, 0, m_data_size, dv.data_ptr(), 0, NULL, NULL), "Error copying input data to device!");
	V_RETURN_CL(clEnqueueWriteBuffer(m_clCommandQueue, m_d_u, CL_FALSE, 0, m_data_size, u.data_ptr(), 0, NULL, NULL), "Error copying input data to device!");
	V_RETURN_CL(clEnqueueWriteBuffer(m_clCommandQueue, m_d_v, CL_FALSE, 0, m_data_size, v.data_ptr(), 0, NULL, NULL), "Error copying input data to device!");
	V_RETURN_CL(clEnqueueWriteBuffer(m_clCommandQueue, m_d_phi, CL_FALSE, 0, m_data_size, du.data_ptr(), 0, NULL, NULL), "Error copying input data to device!");
	V_RETURN_CL(clEnqueueWriteBuffer(m_clCommandQueue, m_d_ksi, CL_FALSE, 0, m_data_size, du.data_ptr(), 0, NULL, NULL), "Error copying input data to device!");

	// wait until all data are copied
	clFinish(m_clCommandQueue);

	// bind kernel arguments (varying during warp levels iterations)
	cl_int cl_error;
	cl_error  = clSetKernelArg(m_clSolverKernel, 6, sizeof(cl_float), (void*)&hx);
	cl_error |= clSetKernelArg(m_clSolverKernel, 7, sizeof(cl_float), (void*)&hy);

	cl_error |= clSetKernelArg(m_clSolverKernel, 12, sizeof(cl_int), (void*)&width);
	cl_error |= clSetKernelArg(m_clSolverKernel, 13, sizeof(cl_int), (void*)&height);
	V_RETURN_CL(cl_error, "Error setting kernel arguments");

	cl_error  = clSetKernelArg(m_clComputePhiKsiKernel, 7, sizeof(cl_float), (void*)&hx);
	cl_error |= clSetKernelArg(m_clComputePhiKsiKernel, 8, sizeof(cl_float), (void*)&hy);

	cl_error |= clSetKernelArg(m_clComputePhiKsiKernel, 11, sizeof(cl_int), (void*)&width);
	cl_error |= clSetKernelArg(m_clComputePhiKsiKernel, 12, sizeof(cl_int), (void*)&height);
	V_RETURN_CL(cl_error, "Error setting kernel arguments");

	size_t globalWorkSize[2] = { GetGlobalWorkSize(width, m_localWorkSize[0]), GetGlobalWorkSize(height, m_localWorkSize[0]) };

	CTimer timer;
	timer.Start();
	// run kernel many times	
	// outer iterations
	for (int i = 0; i < m_solver_iterations; i++) {
		
		// precompute weight values for flow-driven smoothenss
		cl_error  = clSetKernelArg(m_clComputePhiKsiKernel, 2, sizeof(cl_mem), (void*)&m_d_du);
		cl_error |= clSetKernelArg(m_clComputePhiKsiKernel, 3, sizeof(cl_mem), (void*)&m_d_dv);
		V_RETURN_CL(cl_error, "Error setting kernel arguments");

		V_RETURN_CL(clEnqueueNDRangeKernel(m_clCommandQueue, m_clComputePhiKsiKernel, 2, NULL, globalWorkSize, m_localWorkSize, 0, NULL, NULL), "Error executing kernel!");

		// inner iterations
		for (int j = 0; j < m_inner_iterations; j++) {
			// bind input and output buffers
			cl_error  = clSetKernelArg(m_clSolverKernel, 2, sizeof(cl_mem), (void*)&m_d_du);
			cl_error |= clSetKernelArg(m_clSolverKernel, 3, sizeof(cl_mem), (void*)&m_d_dv);

			cl_error |= clSetKernelArg(m_clSolverKernel, 15, sizeof(cl_mem), (void*)&m_d_du_r);
			cl_error |= clSetKernelArg(m_clSolverKernel, 16, sizeof(cl_mem), (void*)&m_d_dv_r);
			V_RETURN_CL(cl_error, "Error setting kernel arguments");

			V_RETURN_CL(clEnqueueNDRangeKernel(m_clCommandQueue, m_clSolverKernel, 2, NULL, globalWorkSize, m_localWorkSize, 0, NULL, NULL), "Error executing kernel!");

			// swap input and output pointers (ping-ponging)
			std::swap(m_d_du, m_d_du_r);
			std::swap(m_d_dv, m_d_dv_r);
		}
	}
	clFinish(m_clCommandQueue);
	timer.Stop();
	std::cout << timer.GetElapsedTime() << std::endl;

	// copy data back to host
	V_RETURN_CL(clEnqueueReadBuffer(m_clCommandQueue, m_d_du, CL_TRUE, 0, m_data_size, du.data_ptr(), 0, NULL, NULL), "Error reading back results from the device!");
	V_RETURN_CL(clEnqueueReadBuffer(m_clCommandQueue, m_d_dv, CL_TRUE, 0, m_data_size, dv.data_ptr(), 0, NULL, NULL), "Error reading back results from the device!");
}