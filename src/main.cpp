#include <iostream>
// Linux declaration
#ifndef _WIN32 
	#include <cmath>
#endif


#include "Image.h"
#include "CTimer.h"
#include "Common.h"

#include "CPUOpticalFlow.h"
#include "GPUNaiveOpticalFlow.h"
#include "GPUOptimizedOpticalFlow.h"
#include "GPUFullOpticalFlow.h"
#include "GPUFlowDrivenRobust.h"

struct Measure
{
	float max;
	float min;
	float mean;
	float sum;
	float value;
};

cl_context			g_CLContext = NULL;
cl_command_queue	g_CLCommandQueue = NULL;
cl_device_id		g_CLDevice = NULL;

bool InitContextResources();
void CleanupContextResources();
Measure EndpointError(const Image& u_field, const Image& v_field, const Image& u_field_gt, const Image& v_field_gt, Image& difference);

int main(int argc, char** argv) 
{
	Image img1;
	Image img2;
	
	Image u_field_gt;
	Image v_field_gt;

	int warp_levels = 15;
	float warp_scale = 0.9f;
	int solver_iterations = 30;
	int inner_iterations = 10;
	float alpha = 4.f;
	float omega = 1.f;
	float e_smooth = 0.001f;
	float e_data = 0.001f;

	if (InitContextResources() &&
		//img1.readImagePGM("./data/my0.pgm") && img2.readImagePGM("./data/my1.pgm")) {
		//u_field_gt.reinit(img1.width(), img1.height(), img1.actual_width(), img1.actual_height(), 0, 0);
		//v_field_gt.reinit(img1.width(), img1.height(), img1.actual_width(), img1.actual_height(), 0, 0);

		img1.readImagePGM("./data/rub1.pgm") && img2.readImagePGM("./data/rub2.pgm") && 
		Image::readMiddlFlowFile("./data/rub_gt.flo", u_field_gt, v_field_gt)) {

		//img1.readImagePGM("./data/frame0.pgm") && img2.readImagePGM("./data/frame1.pgm") &&
		//Image::readMiddlFlowFile("./data/flow10.flo", u_field_gt, v_field_gt)) {

		std::cout << "Initialization: OK" << std::endl;
		std::cout << "Source image size: (" << img1.width() << "x" << img1.height() << ")" << std::endl;

		CTimer timer;
		// result flows, times, measures
		Image u_field_cpu;
		Image v_field_cpu;
		Measure measure_cpu;
		double time_cpu;

		Image u_field_gpu_naive;
		Image v_field_gpu_naive;
		Measure measure_gpu_naive;
		double time_gpu_naive;

		Image u_field_gpu_flow_driven;
		Image v_field_gpu_flow_driven;
		Measure measure_gpu_flow_driven;
		double time_gpu_flow_driven;

		Image u_field_gpu_optimized;
		Image v_field_gpu_optimized;
		Measure measure_gpu_optimized;
		double time_gpu_optimized;

		Image u_field_gpu_full;
		Image v_field_gpu_full;
		Measure measure_gpu_full;
		double time_gpu_full;

		Image difference(img1.width(), img1.height());

		float flow_scale = 2.f * warp_scale;

		Image::saveOpticalFlowRGB(u_field_gt, v_field_gt, flow_scale, "./data/output/flow_gt.pgm");

/* ########################################################################################################################################## */
		std::cout << std::endl << "--- RUN CPU OPTICAL FLOW ---" << std::endl;
		{
			CPUOpticalFlow cpuOpticalFlow(img1, img2, warp_levels, warp_scale, solver_iterations, alpha, omega);
			timer.Start();
			cpuOpticalFlow.computeFlow(u_field_cpu, v_field_cpu);
			timer.Stop();

			time_cpu = timer.GetElapsedTime();
			std::cout << "\nTime:\t" << time_cpu;
			measure_cpu = EndpointError(u_field_cpu, v_field_cpu, u_field_gt, v_field_gt, difference);
			std::cout << "  Mean error:\t" << measure_cpu.mean << "  Max error:\t" << measure_cpu.max << std::endl;
			Image::saveOpticalFlowRGB(u_field_cpu, v_field_cpu, flow_scale, "./data/output/flow_cpu.pgm");
		}
		std::cout << "--- -------------------- ---" << std::endl;

/* ########################################################################################################################################## */
		std::cout << std::endl << "--- RUN GPU NAIVE OPTICAL FLOW ---" << std::endl;
		{
			int localWorkSize[2] = { 32, 16 };
			GPUNaiveOpticalFlow gpuNaiveOpticalFlow(img1, img2, warp_levels, warp_scale, solver_iterations, alpha, omega, 
													g_CLContext, g_CLCommandQueue, localWorkSize);
			if (!gpuNaiveOpticalFlow.initResources(g_CLContext, g_CLDevice)) {
				std::cout << "Error initializing OpenCL resources." << std::endl;
			} else {
				timer.Start();
				gpuNaiveOpticalFlow.computeFlow(u_field_gpu_naive, v_field_gpu_naive);
				timer.Stop();

				time_gpu_naive = timer.GetElapsedTime();
				std::cout << "\nTime:\t" << time_gpu_naive;
				measure_gpu_naive = EndpointError(u_field_gpu_naive, v_field_gpu_naive, u_field_gt, v_field_gt, difference);
				std::cout << "  Mean error:\t" << measure_gpu_naive.mean << "  Max error:\t" << measure_gpu_naive.max << std::endl;
				Image::saveOpticalFlowRGB(u_field_gpu_naive, v_field_gpu_naive, flow_scale, "./data/output/flow_gpu_naive.pgm");
			}
			gpuNaiveOpticalFlow.releaseResources();
		}
		std::cout << "--- -------------------------- ---" << std::endl;

/* ########################################################################################################################################## */
		std::cout << std::endl << "--- RUN GPU FLOW DRIVEN ROBUST OPTICAL FLOW ---" << std::endl;
		{
			int localWorkSize[2] = { 32, 4 };
			GPUFlowDrivenRobust gpuFlowDrivenRobust(img1, img2, warp_levels, warp_scale, solver_iterations, inner_iterations, alpha, omega, e_smooth, e_data,
													g_CLContext, g_CLCommandQueue, localWorkSize);
			if (!gpuFlowDrivenRobust.initResources(g_CLContext, g_CLDevice)) {
				std::cout << "Error initializing OpenCL resources." << std::endl;
			} else {
				timer.Start();
				gpuFlowDrivenRobust.computeFlow(u_field_gpu_flow_driven, v_field_gpu_flow_driven);
				timer.Stop();

				time_gpu_flow_driven = timer.GetElapsedTime();
				std::cout << "\nTime:\t" << time_gpu_flow_driven;
				measure_gpu_flow_driven = EndpointError(u_field_gpu_flow_driven, v_field_gpu_flow_driven, u_field_gt, v_field_gt, difference);
				std::cout << "  Mean error:\t" << measure_gpu_flow_driven.mean << "  Max error:\t" << measure_gpu_flow_driven.max << std::endl;
				Image::saveOpticalFlowRGB(u_field_gpu_flow_driven, v_field_gpu_flow_driven, flow_scale, "./data/output/flow_gpu_flow_driven.pgm");
			}
			gpuFlowDrivenRobust.releaseResources();
		}
		std::cout << "--- --------------------------------------- ---" << std::endl;

/* ########################################################################################################################################## */
		std::cout << std::endl << "--- RUN GPU OPTIMIZED OPTICAL FLOW ---" << std::endl;
		{
			int localWorkSize[2] = { 32, 16 };
			GPUOptimizedOpticalFlow gpuOptimizedOpticalFlow(img1, img2, warp_levels, warp_scale, solver_iterations, alpha, omega,
															g_CLContext, g_CLCommandQueue, localWorkSize);
			if (!gpuOptimizedOpticalFlow.initResources(g_CLContext, g_CLDevice)) {
				std::cout << "Error initializing OpenCL resources." << std::endl;
			} else {
				timer.Start();
				gpuOptimizedOpticalFlow.computeFlow(u_field_gpu_optimized, v_field_gpu_optimized);
				timer.Stop();

				time_gpu_optimized = timer.GetElapsedTime();
				std::cout << "\nTime:\t" << time_gpu_optimized;
				measure_gpu_optimized = EndpointError(u_field_gpu_optimized, v_field_gpu_optimized, u_field_gt, v_field_gt, difference);
				std::cout << "  Mean error:\t" << measure_gpu_optimized.mean << "  Max error:\t" << measure_gpu_optimized.max << std::endl;
				Image::saveOpticalFlowRGB(u_field_gpu_optimized, v_field_gpu_optimized, flow_scale, "./data/output/flow_gpu_optimized.pgm");
			}
			gpuOptimizedOpticalFlow.releaseResources();

		}
		std::cout << "--- ------------------------------ ---" << std::endl;

/* ########################################################################################################################################## */
		std::cout << std::endl << "--- RUN GPU FULL OPTICAL FLOW ---" << std::endl;
		{
			int localWorkSize[2] = { 32, 4 };
			GPUFullOpticalFlow gpuFullOpticalFlow(img1, img2, warp_levels, warp_scale, solver_iterations, alpha, omega,
												  g_CLContext, g_CLCommandQueue, localWorkSize);
			if (!gpuFullOpticalFlow.initResources(g_CLContext, g_CLDevice)) {
				std::cout << "Error initializing OpenCL resources." << std::endl;
			} else {
				timer.Start();
				gpuFullOpticalFlow.computeFlow(u_field_gpu_full, v_field_gpu_full);
				timer.Stop();

				time_gpu_full = timer.GetElapsedTime();
				std::cout << "\nTime:\t" << time_gpu_full;
				measure_gpu_full = EndpointError(u_field_gpu_full, v_field_gpu_full, u_field_gt, v_field_gt, difference);
				std::cout << "  Mean error:\t" << measure_gpu_full.mean << "  Max error:\t" << measure_gpu_full.max << std::endl;
				Image::saveOpticalFlowRGB(u_field_gpu_full, v_field_gpu_full, flow_scale, "./data/output/flow_gpu_full.pgm");
			}
			gpuFullOpticalFlow.releaseResources();

		}
		std::cout << "--- ------------------------- ---" << std::endl;

/* ########################################################################################################################################## */
		std::cout << std::endl << "*************** METHODS COMPARISON ***************" << std::endl << std::endl;
		{
			std::cout << "Method\t\tTime\t\tMean error\tMax error\tSpeed-up" << std::endl;
			std::cout << "CPU\t\t" << time_cpu << "\t\t" << measure_cpu.mean << "\t" << measure_cpu.max << "\t\t1.0" << std::endl;

			std::cout << "GPU Naive\t" << time_gpu_naive << "\t\t" << measure_gpu_naive.mean << "\t" << measure_gpu_naive.max << "\t\t" << time_cpu / time_gpu_naive << std::endl;

			std::cout << "GPU Optimized\t" << time_gpu_optimized << "\t\t" << measure_gpu_optimized.mean << "\t" << measure_gpu_optimized.max << "\t\t" << time_cpu / time_gpu_optimized << std::endl;

			std::cout << "GPU Full\t" << time_gpu_full << "\t\t" << measure_gpu_full.mean << "\t" << measure_gpu_full.max << "\t\t" << time_cpu / time_gpu_full << std::endl;
		}
		std::cout << "*************** ****************** ***************" << std::endl;

	}
	CleanupContextResources();

	std::cout << "Press Enter to continue";
	std::getchar();
	return 0;
}

/**
* Compute Endpoint error (EE) between ground truth result and computed flow field
*
* (u/v)_field		Computed result
* (u/v)_field_gt	Ground truth result
* difference		Endpoint error result as an Image
*/
Measure EndpointError(const Image& u_field, const Image& v_field, const Image& u_field_gt, const Image& v_field_gt, Image& difference)
{

	if ((u_field.actual_width() != u_field_gt.actual_width()) || (u_field.actual_height() != u_field.actual_height()))
		return Measure();

	int width = u_field.actual_width();
	int height = u_field.actual_height();

	float endpointError;
	float max = 0;

	// Number of pixels considered to be a valid flow result
	int countPixels = 0;
	float sum = 0;

	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {

			/* unknown flow */
			if ((std::fabs(u_field_gt.pixel_r(x, y)) > 1e6) || (std::fabs(v_field_gt.pixel_r(x, y)) > 1e6) || x != x || y != y) {
				difference.pixel_w(x, y) = 0.f;
			} else {
				endpointError = sqrt((u_field.pixel_r(x, y) - u_field_gt.pixel_r(x, y)) * 
									 (u_field.pixel_r(x, y) - u_field_gt.pixel_r(x, y)) +
									 (v_field.pixel_r(x, y) - v_field_gt.pixel_r(x, y)) *
									 (v_field.pixel_r(x, y) - v_field_gt.pixel_r(x, y)));
				difference.pixel_w(x, y) = endpointError;

				if (endpointError > max)
					max = endpointError;

				sum = sum + endpointError;
				countPixels++;
			}
		}
	}
	// Save current statistics measures
	Measure m;
	m.max = max;
	m.sum = sum;
	m.mean = sum / (float)countPixels;

	return m;
}	

bool InitContextResources()
{
	//error code
	cl_int clError;
	cl_platform_id		g_CLPlatform[2];

	//get platform ID
	V_RETURN_FALSE_CL(clGetPlatformIDs(2, g_CLPlatform, NULL), "Failed to get CL platform ID");

	//get a reference to the first available GPU device
	#ifdef _WIN32   // Windows version
		V_RETURN_FALSE_CL(clGetDeviceIDs(g_CLPlatform[1], CL_DEVICE_TYPE_GPU, 1, &g_CLDevice, NULL), "No GPU device found.");
	#else           // Linux version
		V_RETURN_FALSE_CL(clGetDeviceIDs(g_CLPlatform[0], CL_DEVICE_TYPE_GPU, 1, &g_CLDevice, NULL), "No GPU device found.");
	#endif

	char deviceName[256];
	V_RETURN_FALSE_CL(clGetDeviceInfo(g_CLDevice, CL_DEVICE_NAME, 256, &deviceName, NULL), "Unable to query device name.");
	cout << "Device: " << deviceName << endl;

	//Create a new OpenCL context on the selected device
	g_CLContext = clCreateContext(0, 1, &g_CLDevice, NULL, NULL, &clError);
	V_RETURN_FALSE_CL(clError, "Failed to create OpenCL context.");

	//Finally, create the command queue. All the asynchronous commands to the device will be issued
	//from the CPU into this queue. This way the host program can continue the execution until some results
	//from that device are needed.
	g_CLCommandQueue = clCreateCommandQueue(g_CLContext, g_CLDevice, 0, &clError);
	V_RETURN_FALSE_CL(clError, "Failed to create the command queue in the context");

	return true;
}

void CleanupContextResources()
{
	if (g_CLCommandQueue)	clReleaseCommandQueue(g_CLCommandQueue);
	if (g_CLContext)			clReleaseContext(g_CLContext);
}
