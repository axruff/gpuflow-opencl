#include "GPUFullOpticalFlow.h"

#include <algorithm>

GPUFullOpticalFlow::GPUFullOpticalFlow(const Image& img1, const Image& img2, int warp_levels, float warp_scale, int solver_iterations, float alpha, float omega,
	cl_context clContext, cl_command_queue clCommandQueue, int localWorkSize[2])
	: OpticalFlowBase(img1, img2, warp_levels, warp_scale, solver_iterations, alpha, omega),
	m_clContext(clContext), m_clCommandQueue(clCommandQueue),
	m_clProgram(NULL), m_clSolverKernel(NULL), m_clZeroKernel(NULL), m_clAddKernel(NULL),
	m_clReflectHorizontalBoudariesKernel(NULL), m_clReflectVerticalBoudariesKernel(NULL),
	m_clResampleXKernel(NULL), m_clResampleYKernel(NULL),
	m_d_src_Img1(NULL), m_d_src_Img2(NULL), m_d_Img_2_br(NULL),
	m_d_Img_1(NULL), m_d_Img_2(NULL), m_d_du(NULL), m_d_dv(NULL), m_d_u(NULL), m_d_v(NULL),
	m_data_size(0)
{
	m_localWorkSize[0] = localWorkSize[0];
	m_localWorkSize[1] = localWorkSize[1];
}

GPUFullOpticalFlow::~GPUFullOpticalFlow()
{
}

bool GPUFullOpticalFlow::initResources(cl_context context, cl_device_id device)
{
	cl_int cl_error;
	char * program_code;
	size_t program_size;
	
	LoadProgram("./src/kernels/FullGPUSolver.cl", &program_code, &program_size);

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

	m_clZeroKernel = clCreateKernel(m_clProgram, "Zero", &cl_error);
	V_RETURN_FALSE_CL(cl_error, "Failed to create kernel.");

	m_clAddKernel = clCreateKernel(m_clProgram, "Add", &cl_error);
	V_RETURN_FALSE_CL(cl_error, "Failed to create kernel.");

	m_clBackwardRegistrationKernel = clCreateKernel(m_clProgram, "BackwardRegistration", &cl_error);
	V_RETURN_FALSE_CL(cl_error, "Failed to create kernel.");

	m_clReflectHorizontalBoudariesKernel = clCreateKernel(m_clProgram, "ReflectHorizontalBoudaries", &cl_error);
	V_RETURN_FALSE_CL(cl_error, "Failed to create kernel.");	
	
	m_clReflectVerticalBoudariesKernel = clCreateKernel(m_clProgram, "ReflectVerticalBoudaries", &cl_error);
	V_RETURN_FALSE_CL(cl_error, "Failed to create kernel.");

	m_clResampleXKernel = clCreateKernel(m_clProgram, "ResampleX", &cl_error);
	V_RETURN_FALSE_CL(cl_error, "Failed to create kernel.");

	m_clResampleYKernel = clCreateKernel(m_clProgram, "ResampleY", &cl_error);
	V_RETURN_FALSE_CL(cl_error, "Failed to create kernel.");

	// create device resources
	int bx = 1;
	int by = 1;
	// temporal image to get right image sizes
	Image img(m_source_img_1.width(), m_source_img_1.height(), bx, by);
	int height = img.height();
	int pitch = img.pitch();
	m_data_size = pitch * (height + 2 * by) * sizeof(cl_float);

	m_d_src_Img1 = clCreateBuffer(context, CL_MEM_READ_ONLY, m_data_size, NULL, &cl_error);
	V_RETURN_FALSE_CL(cl_error, "Error allocating device memory");
	m_d_src_Img2 = clCreateBuffer(context, CL_MEM_READ_ONLY, m_data_size, NULL, &cl_error);
	V_RETURN_FALSE_CL(cl_error, "Error allocating device memory");

	
	m_d_Img_1 = clCreateBuffer(context, CL_MEM_READ_WRITE, m_data_size, NULL, &cl_error);
	V_RETURN_FALSE_CL(cl_error, "Error allocating device memory");
	m_d_Img_2 = clCreateBuffer(context, CL_MEM_READ_WRITE, m_data_size, NULL, &cl_error);
	V_RETURN_FALSE_CL(cl_error, "Error allocating device memory");
	m_d_Img_2_br = clCreateBuffer(context, CL_MEM_READ_WRITE, m_data_size, NULL, &cl_error);
	V_RETURN_FALSE_CL(cl_error, "Error allocating device memory");
	m_d_u = clCreateBuffer(context, CL_MEM_READ_WRITE, m_data_size, NULL, &cl_error);
	V_RETURN_FALSE_CL(cl_error, "Error allocating device memory");
	m_d_v = clCreateBuffer(context, CL_MEM_READ_WRITE, m_data_size, NULL, &cl_error);
	V_RETURN_FALSE_CL(cl_error, "Error allocating device memory");
	m_d_du = clCreateBuffer(context, CL_MEM_READ_WRITE, m_data_size, NULL, &cl_error);
	V_RETURN_FALSE_CL(cl_error, "Error allocating device memory");
	m_d_dv = clCreateBuffer(context, CL_MEM_READ_WRITE, m_data_size, NULL, &cl_error);
	V_RETURN_FALSE_CL(cl_error, "Error allocating device memory");
	m_d_du_r = clCreateBuffer(context, CL_MEM_READ_WRITE, m_data_size, NULL, &cl_error);
	V_RETURN_FALSE_CL(cl_error, "Error allocating device memory");
	m_d_dv_r = clCreateBuffer(context, CL_MEM_READ_WRITE, m_data_size, NULL, &cl_error);
	V_RETURN_FALSE_CL(cl_error, "Error allocating device memory");

	// bind kernel arguments (constant for all iterations)
	/* SolverKernel */
	cl_error |= clSetKernelArg(m_clSolverKernel, 1, sizeof(cl_mem), (void*)&m_d_Img_2_br);

	cl_error |= clSetKernelArg(m_clSolverKernel, 8, sizeof(cl_float), (void*)&m_alpha);
	cl_error |= clSetKernelArg(m_clSolverKernel, 9, sizeof(cl_float), (void*)&m_omega);
	cl_error |= clSetKernelArg(m_clSolverKernel, 10, sizeof(cl_int), (void*)&bx);
	cl_error |= clSetKernelArg(m_clSolverKernel, 11, sizeof(cl_int), (void*)&by);

	cl_error |= clSetKernelArg(m_clSolverKernel, 14, sizeof(cl_int), (void*)&pitch);
	V_RETURN_FALSE_CL(cl_error, "Error setting kernel arguments");

	/* BackwardRegistrationKernel */
	cl_error |= clSetKernelArg(m_clBackwardRegistrationKernel, 6, sizeof(cl_int), (void*)&bx);
	cl_error |= clSetKernelArg(m_clBackwardRegistrationKernel, 7, sizeof(cl_int), (void*)&by);
	
	cl_error |= clSetKernelArg(m_clBackwardRegistrationKernel, 10, sizeof(cl_int), (void*)&pitch);
	cl_error |= clSetKernelArg(m_clBackwardRegistrationKernel, 11, sizeof(cl_mem), (void*)&m_d_Img_2_br);
	V_RETURN_FALSE_CL(cl_error, "Error setting kernel arguments");

	/* ReflectHorizontalBoudaries and ReflectVerticalBoudaries */
	cl_error  = clSetKernelArg(m_clReflectHorizontalBoudariesKernel, 1, sizeof(cl_int), (void*)&bx);
	cl_error |= clSetKernelArg(m_clReflectHorizontalBoudariesKernel, 2, sizeof(cl_int), (void*)&by);
	cl_error |= clSetKernelArg(m_clReflectHorizontalBoudariesKernel, 5, sizeof(cl_int), (void*)&pitch);
	
	cl_error |= clSetKernelArg(m_clReflectVerticalBoudariesKernel,	 1, sizeof(cl_int), (void*)&bx);
	cl_error |= clSetKernelArg(m_clReflectVerticalBoudariesKernel,	 2, sizeof(cl_int), (void*)&by);
	cl_error |= clSetKernelArg(m_clReflectVerticalBoudariesKernel,	 5, sizeof(cl_int), (void*)&pitch);
	V_RETURN_FALSE_CL(cl_error, "Error setting kernel arguments");

	/* ResampleX and ResampleY */
	cl_error  = clSetKernelArg(m_clResampleXKernel, 5, sizeof(cl_int), (void*)&pitch);
	cl_error |= clSetKernelArg(m_clResampleYKernel, 5, sizeof(cl_int), (void*)&pitch);
	V_RETURN_FALSE_CL(cl_error, "Error setting kernel arguments");


	return true;
}

void GPUFullOpticalFlow::releaseResources()
{
	SAFE_RELEASE_MEMOBJECT(m_d_src_Img1);
	SAFE_RELEASE_MEMOBJECT(m_d_src_Img2);
	SAFE_RELEASE_MEMOBJECT(m_d_Img_1);
	SAFE_RELEASE_MEMOBJECT(m_d_Img_2);
	SAFE_RELEASE_MEMOBJECT(m_d_Img_2_br);
	SAFE_RELEASE_MEMOBJECT(m_d_du);
	SAFE_RELEASE_MEMOBJECT(m_d_dv);
	SAFE_RELEASE_MEMOBJECT(m_d_du_r);
	SAFE_RELEASE_MEMOBJECT(m_d_dv_r);
	SAFE_RELEASE_MEMOBJECT(m_d_u);
	SAFE_RELEASE_MEMOBJECT(m_d_v);

	SAFE_RELEASE_KERNEL(m_clZeroKernel);
	SAFE_RELEASE_KERNEL(m_clAddKernel);
	SAFE_RELEASE_KERNEL(m_clSolverKernel);
	SAFE_RELEASE_KERNEL(m_clBackwardRegistrationKernel);
	SAFE_RELEASE_KERNEL(m_clReflectHorizontalBoudariesKernel);
	SAFE_RELEASE_KERNEL(m_clReflectVerticalBoudariesKernel);
	SAFE_RELEASE_KERNEL(m_clResampleXKernel);
	SAFE_RELEASE_KERNEL(m_clResampleYKernel);
	SAFE_RELEASE_PROGRAM(m_clProgram);
}

void GPUFullOpticalFlow::computeFlow(Image& u, Image& v)
{
	int source_width;	// size in x-direction(source image resolution)
	int source_height;	// size in y-direction(source image resolution)
	int level_width;	// size in x-direction(current resolution)
	int level_height;	// size in y-direction(current resolution)
	int prev_width;		// size in x-direction(previous resolution)
	int prev_height;	// size in y-direction(previous resolution)
	float hx;			// spacing in x-direction (current resolution) 
	float hy;			// spacing in y-direction (current resolution) 

	
	source_width = m_source_img_1.width();
	source_height = m_source_img_1.height();
	
	int current_warp_level = std::min(m_warp_levels, computeMaxWarpLevels()) - 1;

	// initialize output flow arrays and copy source images to device
	u.reinit(source_width, source_height, source_width, source_height, 1, 1);
	u = m_source_img_1;
	V_RETURN_CL(clEnqueueWriteBuffer(m_clCommandQueue, m_d_src_Img1, CL_FALSE, 0, m_data_size, u.data_ptr(), 0, NULL, NULL), "Error copying input data to device!");
	v.reinit(source_width, source_height, source_width, source_height, 1, 1);
	v = m_source_img_2;
	V_RETURN_CL(clEnqueueWriteBuffer(m_clCommandQueue, m_d_src_Img2, CL_FALSE, 0, m_data_size, v.data_ptr(), 0, NULL, NULL), "Error copying input data to device!");

	prev_width = 0;
	prev_height = 0;
	clFinish(m_clCommandQueue);

	while (current_warp_level >= 0) {
		// compute level sizes
		level_width = static_cast<int>(ceil(source_width * pow(m_warp_scale, current_warp_level)));
		level_height = static_cast<int>(ceil(source_height * pow(m_warp_scale, current_warp_level)));
		hx = source_width / static_cast<float>(level_width);
		hy = source_height / static_cast<float>(level_height);

		std::cout << "Solve level: " << current_warp_level << " (" << level_width << "x" << level_height << ")" << std::endl;
	
		// GPU area based resampling
		// src		    : in
		// dst			: out
		// m_d_Img_2_br : temporary buffer
		// image resampling
		if (current_warp_level == 0) {
			std::swap(m_d_Img_1, m_d_src_Img1);
			std::swap(m_d_Img_2, m_d_src_Img2);
		} else {
			resampleAreaBased(m_d_src_Img1, m_d_Img_1, source_width, source_height, level_width, level_height);
			resampleAreaBased(m_d_src_Img2, m_d_Img_2, source_width, source_height, level_width, level_height);
		}

		// displacement field resampling
		if (prev_width == 0) {
			// first iteration, initialize with zeros
			zeroDeviceBuffer(m_d_u);
			zeroDeviceBuffer(m_d_v);
		} else {
			resampleAreaBased(m_d_u, m_d_du, prev_width, prev_height, level_width, level_height);
			resampleAreaBased(m_d_v, m_d_dv, prev_width, prev_height, level_width, level_height);
			std::swap(m_d_u, m_d_du);
			std::swap(m_d_v, m_d_dv);
		}

		// perform backward registration
		// m_d_Img_1	: in
		// m_d_Img_2	: in
		// m_d_u		: in
		// m_d_v		: in
		// m_d_Img_2_br	: out
		backwardRegistration(hx, hy, level_width, level_height);
	
		// reflect bouundaries
		// m_d_Img_1	: in:out
		// m_d_Img_2_br	: in:out
		reflectBoudaries(level_width, level_height);

		// solve difference problem at current resolution to obtain increment
		// m_d_Img_1	: in
		// m_d_Img_2_br	: in
		// m_d_u		: in
		// m_d_v		: in
		// m_d_du		: in:out
		// m_d_dv		: in:out
		solveDifference(hx, hy, level_width, level_height);

		// add solved increment to the global flow
		// m_d_u		: in:out
		// m_d_v		: in:out
		// m_d_du		: in
		// m_d_dv		: in
		addFlowIncrement();
		//u += du;
		//v += dv;

		// go to the next level
		prev_width = level_width;
		prev_height = level_height;
		current_warp_level--;
	}
	// copy data back to host
	V_RETURN_CL(clEnqueueReadBuffer(m_clCommandQueue, m_d_u, CL_TRUE, 0, m_data_size, u.data_ptr(), 0, NULL, NULL), "Error reading back results from the device!");
	V_RETURN_CL(clEnqueueReadBuffer(m_clCommandQueue, m_d_v, CL_TRUE, 0, m_data_size, v.data_ptr(), 0, NULL, NULL), "Error reading back results from the device!");
}

void GPUFullOpticalFlow::solveDifference(float hx, float hy, int width, int height)
{
	// we run Zero kernel to initialize du, dv, du_r and dv_r with zeros
	size_t globalWorkSizeZeroKernel = m_data_size / sizeof(float);
	zeroDeviceBuffer(m_d_du);
	zeroDeviceBuffer(m_d_dv);
	zeroDeviceBuffer(m_d_du_r);
	zeroDeviceBuffer(m_d_dv_r);

	// wait until all data are initialized
	clFinish(m_clCommandQueue);

	// bind kernel arguments (varying during warp levels iterations)
	cl_int cl_error;
	cl_error  = clSetKernelArg(m_clSolverKernel, 0, sizeof(cl_mem), (void*)&m_d_Img_1);

	cl_error |= clSetKernelArg(m_clSolverKernel, 4, sizeof(cl_mem), (void*)&m_d_u);
	cl_error |= clSetKernelArg(m_clSolverKernel, 5, sizeof(cl_mem), (void*)&m_d_v);
	cl_error |= clSetKernelArg(m_clSolverKernel, 6, sizeof(cl_float), (void*)&hx);
	cl_error |= clSetKernelArg(m_clSolverKernel, 7, sizeof(cl_float), (void*)&hy);

	cl_error |= clSetKernelArg(m_clSolverKernel, 12, sizeof(cl_int), (void*)&width);
	cl_error |= clSetKernelArg(m_clSolverKernel, 13, sizeof(cl_int), (void*)&height);
	V_RETURN_CL(cl_error, "Error setting kernel arguments");

	size_t globalWorkSize[2] = { GetGlobalWorkSize(width, m_localWorkSize[0]), GetGlobalWorkSize(height, m_localWorkSize[0]) };

	// run kernel many times	
	for (int i = 0; i < m_solver_iterations; i++) {
		// bind input and output buffers
		cl_error  = clSetKernelArg(m_clSolverKernel, 2, sizeof(cl_mem), (void*)&m_d_du);
		cl_error |= clSetKernelArg(m_clSolverKernel, 3, sizeof(cl_mem), (void*)&m_d_dv);

		cl_error |= clSetKernelArg(m_clSolverKernel, 15, sizeof(cl_mem), (void*)&m_d_du_r);
		cl_error |= clSetKernelArg(m_clSolverKernel, 16, sizeof(cl_mem), (void*)&m_d_dv_r);
		V_RETURN_CL(cl_error, "Error setting kernel arguments");

		V_RETURN_CL(clEnqueueNDRangeKernel(m_clCommandQueue, m_clSolverKernel , 2, NULL, globalWorkSize, m_localWorkSize, 0, NULL, NULL), "Error executing kernel!");

		// swap input and output pointers (ping-ponging)
		std::swap(m_d_du, m_d_du_r);
		std::swap(m_d_dv, m_d_dv_r);
	}
	clFinish(m_clCommandQueue);
}

void GPUFullOpticalFlow::backwardRegistration(float hx, float hy, int width, int height)
{
	cl_int cl_error;

	// reflect img_2 boudaries for correct backward registration on borders
	cl_error  = clSetKernelArg(m_clReflectHorizontalBoudariesKernel, 3, sizeof(cl_int), (void*)&width);
	cl_error |= clSetKernelArg(m_clReflectHorizontalBoudariesKernel, 4, sizeof(cl_int), (void*)&height);

	cl_error |= clSetKernelArg(m_clReflectVerticalBoudariesKernel, 3, sizeof(cl_int), (void*)&width);
	cl_error |= clSetKernelArg(m_clReflectVerticalBoudariesKernel, 4, sizeof(cl_int), (void*)&height);

	cl_error |= clSetKernelArg(m_clReflectHorizontalBoudariesKernel, 0, sizeof(cl_mem), (void*)&m_d_Img_2);
	cl_error |= clSetKernelArg(m_clReflectVerticalBoudariesKernel, 0, sizeof(cl_mem), (void*)&m_d_Img_2);
	V_RETURN_CL(cl_error, "Error setting kernel arguments");


	size_t globalWorkSize1D = GetGlobalWorkSize(width, m_localWorkSize[0]);
	V_RETURN_CL(clEnqueueNDRangeKernel(m_clCommandQueue, m_clReflectHorizontalBoudariesKernel, 1, NULL, &globalWorkSize1D, &m_localWorkSize[0], 0, NULL, NULL), "Error executing kernel!");
	globalWorkSize1D = GetGlobalWorkSize(height, m_localWorkSize[0]);
	V_RETURN_CL(clEnqueueNDRangeKernel(m_clCommandQueue, m_clReflectVerticalBoudariesKernel, 1, NULL, &globalWorkSize1D, &m_localWorkSize[0], 0, NULL, NULL), "Error executing kernel!");
   
	// run backward registration kernel
	cl_error  = clSetKernelArg(m_clBackwardRegistrationKernel, 0, sizeof(cl_mem), (void*)&m_d_Img_1);
	cl_error |= clSetKernelArg(m_clBackwardRegistrationKernel, 1, sizeof(cl_mem), (void*)&m_d_Img_2);
	cl_error |= clSetKernelArg(m_clBackwardRegistrationKernel, 2, sizeof(cl_mem), (void*)&m_d_u);
	cl_error |= clSetKernelArg(m_clBackwardRegistrationKernel, 3, sizeof(cl_mem), (void*)&m_d_v);
	cl_error |= clSetKernelArg(m_clBackwardRegistrationKernel, 4, sizeof(cl_float), (void*)&hx);
	cl_error |= clSetKernelArg(m_clBackwardRegistrationKernel, 5, sizeof(cl_float), (void*)&hy);

	cl_error |= clSetKernelArg(m_clBackwardRegistrationKernel, 8, sizeof(cl_int), (void*)&width);
	cl_error |= clSetKernelArg(m_clBackwardRegistrationKernel, 9, sizeof(cl_int), (void*)&height);
	V_RETURN_CL(cl_error, "Error setting kernel arguments");

	size_t globalWorkSize[2] = { GetGlobalWorkSize(width, m_localWorkSize[0]), GetGlobalWorkSize(height, m_localWorkSize[0]) };

	V_RETURN_CL(clEnqueueNDRangeKernel(m_clCommandQueue, m_clBackwardRegistrationKernel, 2, NULL, globalWorkSize, m_localWorkSize, 0, NULL, NULL), "Error executing kernel!");
	
	clFinish(m_clCommandQueue);
}

void GPUFullOpticalFlow::reflectBoudaries(int width, int height)
{
	cl_int cl_error;
	size_t globalWorkSize1D;

	cl_error  = clSetKernelArg(m_clReflectHorizontalBoudariesKernel, 3, sizeof(cl_int), (void*)&width);
	cl_error |= clSetKernelArg(m_clReflectHorizontalBoudariesKernel, 4, sizeof(cl_int), (void*)&height);

	cl_error |= clSetKernelArg(m_clReflectVerticalBoudariesKernel,   3, sizeof(cl_int), (void*)&width);
	cl_error |= clSetKernelArg(m_clReflectVerticalBoudariesKernel,   4, sizeof(cl_int), (void*)&height);
	V_RETURN_CL(cl_error, "Error setting kernel arguments");

	V_RETURN_CL(clSetKernelArg(m_clReflectHorizontalBoudariesKernel, 0, sizeof(cl_mem), (void*)&m_d_Img_1), "Error setting kernel arguments");
	V_RETURN_CL(clSetKernelArg(m_clReflectVerticalBoudariesKernel,   0, sizeof(cl_mem), (void*)&m_d_Img_1), "Error setting kernel arguments");

	globalWorkSize1D = GetGlobalWorkSize(width, m_localWorkSize[0]);
	V_RETURN_CL(clEnqueueNDRangeKernel(m_clCommandQueue, m_clReflectHorizontalBoudariesKernel, 1, NULL, &globalWorkSize1D, &m_localWorkSize[0], 0, NULL, NULL), "Error executing kernel!");
	globalWorkSize1D = GetGlobalWorkSize(height, m_localWorkSize[0]);
	V_RETURN_CL(clEnqueueNDRangeKernel(m_clCommandQueue, m_clReflectVerticalBoudariesKernel,   1, NULL, &globalWorkSize1D, &m_localWorkSize[0], 0, NULL, NULL), "Error executing kernel!");

	V_RETURN_CL(clSetKernelArg(m_clReflectHorizontalBoudariesKernel, 0, sizeof(cl_mem), (void*)&m_d_Img_2_br), "Error setting kernel arguments");
	V_RETURN_CL(clSetKernelArg(m_clReflectVerticalBoudariesKernel,   0, sizeof(cl_mem), (void*)&m_d_Img_2_br), "Error setting kernel arguments");
	globalWorkSize1D = GetGlobalWorkSize(width, m_localWorkSize[0]);
	V_RETURN_CL(clEnqueueNDRangeKernel(m_clCommandQueue, m_clReflectHorizontalBoudariesKernel, 1, NULL, &globalWorkSize1D, &m_localWorkSize[0], 0, NULL, NULL), "Error executing kernel!");
	globalWorkSize1D = GetGlobalWorkSize(height, m_localWorkSize[0]);
	V_RETURN_CL(clEnqueueNDRangeKernel(m_clCommandQueue, m_clReflectVerticalBoudariesKernel,   1, NULL, &globalWorkSize1D, &m_localWorkSize[0], 0, NULL, NULL), "Error executing kernel!");

	clFinish(m_clCommandQueue);
}

void GPUFullOpticalFlow::addFlowIncrement()
{
	cl_int cl_error;
	size_t globalWorkSizeAddKernel = m_data_size / sizeof(float) / 4;
	
	// u += du
	cl_error  = clSetKernelArg(m_clAddKernel, 0, sizeof(cl_mem), (void*)&m_d_u);
	cl_error |= clSetKernelArg(m_clAddKernel, 1, sizeof(cl_mem), (void*)&m_d_du);
	V_RETURN_CL(cl_error, "Error setting kernel arguments");
	V_RETURN_CL(clEnqueueNDRangeKernel(m_clCommandQueue, m_clAddKernel, 1, NULL, &globalWorkSizeAddKernel, NULL, 0, NULL, NULL), "Error executing kernel!");

	// v += dv
	cl_error  = clSetKernelArg(m_clAddKernel, 0, sizeof(cl_mem), (void*)&m_d_v);
	cl_error |= clSetKernelArg(m_clAddKernel, 1, sizeof(cl_mem), (void*)&m_d_dv);
	V_RETURN_CL(cl_error, "Error setting kernel arguments");
	V_RETURN_CL(clEnqueueNDRangeKernel(m_clCommandQueue, m_clAddKernel, 1, NULL, &globalWorkSizeAddKernel, NULL, 0, NULL, NULL), "Error executing kernel!");

	clFinish(m_clCommandQueue);
}

void GPUFullOpticalFlow::resample_x(cl_mem src, cl_mem dst, int src_width, int src_height, int dst_width, int dst_height)
{
	cl_int cl_error;
	size_t globalWorkSize = GetGlobalWorkSize(src_height, m_localWorkSize[0]);

	cl_error  = clSetKernelArg(m_clResampleXKernel, 0, sizeof(cl_mem), (void*)&src);
	cl_error |= clSetKernelArg(m_clResampleXKernel, 1, sizeof(cl_mem), (void*)&dst);
	cl_error |= clSetKernelArg(m_clResampleXKernel, 2, sizeof(cl_int), (void*)&src_height);
	cl_error |= clSetKernelArg(m_clResampleXKernel, 3, sizeof(cl_int), (void*)&src_width);
	cl_error |= clSetKernelArg(m_clResampleXKernel, 4, sizeof(cl_int), (void*)&dst_width);
	V_RETURN_CL(cl_error, "Error setting kernel arguments");
	V_RETURN_CL(clEnqueueNDRangeKernel(m_clCommandQueue, m_clResampleXKernel, 1, NULL, &globalWorkSize, &m_localWorkSize[0], 0, NULL, NULL), "Error executing kernel!");
}

void GPUFullOpticalFlow::resample_y(cl_mem src, cl_mem dst, int src_width, int src_height, int dst_width, int dst_height)
{
	cl_int cl_error;
	size_t globalWorkSize = GetGlobalWorkSize(src_width, m_localWorkSize[0]);

	cl_error  = clSetKernelArg(m_clResampleYKernel, 0, sizeof(cl_mem), (void*)&src);
	cl_error |= clSetKernelArg(m_clResampleYKernel, 1, sizeof(cl_mem), (void*)&dst);
	cl_error |= clSetKernelArg(m_clResampleYKernel, 2, sizeof(cl_int), (void*)&src_width);
	cl_error |= clSetKernelArg(m_clResampleYKernel, 3, sizeof(cl_int), (void*)&src_height);
	cl_error |= clSetKernelArg(m_clResampleYKernel, 4, sizeof(cl_int), (void*)&dst_height);
	V_RETURN_CL(cl_error, "Error setting kernel arguments");
	V_RETURN_CL(clEnqueueNDRangeKernel(m_clCommandQueue, m_clResampleYKernel, 1, NULL, &globalWorkSize, &m_localWorkSize[0], 0, NULL, NULL), "Error executing kernel!");
}

void GPUFullOpticalFlow::resampleAreaBased(cl_mem src, cl_mem dst, int src_width, int src_height, int dst_width, int dst_height)
{
	/* if interpolation */
	if (dst_height >= src_height) {
		resample_x(src, m_d_Img_2_br, src_width, src_height, dst_width, src_height);
		resample_y(m_d_Img_2_br, dst, dst_width, src_height, dst_width, dst_height);
	}
	/* if restriction */
	else {
		resample_y(src, m_d_Img_2_br, src_width, src_height, src_width, dst_height);
		resample_x(m_d_Img_2_br, dst, src_width, dst_height, dst_width, dst_height);
	}
	clFinish(m_clCommandQueue);
}

void GPUFullOpticalFlow::zeroDeviceBuffer(cl_mem mem)
{
	size_t globalWorkSizeZeroKernel = m_data_size / sizeof(float);
	V_RETURN_CL(clSetKernelArg(m_clZeroKernel, 0, sizeof(cl_mem), (void*)&mem), "Error setting kernel arguments");
	V_RETURN_CL(clEnqueueNDRangeKernel(m_clCommandQueue, m_clZeroKernel, 1, NULL, &globalWorkSizeZeroKernel, NULL, 0, NULL, NULL), "Error executing kernel!");
}