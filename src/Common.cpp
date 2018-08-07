#include "Common.h"
#include "CTimer.h"

void PrintBuildLog(cl_program Program, cl_device_id Device)
{
	cl_build_status buildStatus;
	clGetProgramBuildInfo(Program, Device, CL_PROGRAM_BUILD_STATUS, sizeof(cl_build_status), &buildStatus, NULL);
	if(buildStatus == CL_SUCCESS)
		return;

	//there were some errors.
	char* buildLog;
	size_t logSize;
	clGetProgramBuildInfo(Program, Device, CL_PROGRAM_BUILD_LOG, 0, NULL, &logSize);
	buildLog = new char[logSize + 1];

	clGetProgramBuildInfo(Program, Device, CL_PROGRAM_BUILD_LOG, logSize, buildLog, NULL);
	buildLog[logSize] = '\0';

	cout<<"There were build errors:"<<endl;
	cout<<buildLog<<endl;

	delete [] buildLog;
}

void LoadProgram(const char* Path, char** pSource, size_t* SourceSize)
{
	FILE* pFileStream = NULL;

	// open the OpenCL source code file
	#ifdef _WIN32   // Windows version
		if(fopen_s(&pFileStream, Path, "rb") != 0) 
		{       
			cout<<"File not found: "<<Path;
			return;
		}
	#else           // Linux version
		pFileStream = fopen(Path, "rb");
		if(pFileStream == 0) 
		{       
			cout<<"File not found: "<<Path;
			return;
		}
	#endif

	//get the length of the source code
	fseek(pFileStream, 0, SEEK_END);
	*SourceSize = ftell(pFileStream);
	fseek(pFileStream, 0, SEEK_SET);

	*pSource = new char[*SourceSize + 1];
	fread(*pSource, *SourceSize, 1, pFileStream);
	fclose(pFileStream);
	(*pSource)[*SourceSize] = '\0';
}

size_t GetGlobalWorkSize(size_t DataSize, size_t LocalWorkSize)
{
	size_t r = DataSize % LocalWorkSize;
	if(r == 0)
	{
		return DataSize;
	}
	else
	{
		return DataSize + LocalWorkSize - r;
	}
}


double RunKernelNTimes(cl_command_queue CommandQueue, cl_kernel Kernel, cl_uint Dimensions, const size_t* pGlobalWorkSize,
						const size_t* pLocalWorkSize, unsigned int NIterations)
{
	CTimer timer;
	cl_int clErr;

	//wait until the command queue is empty... iniefficient but allows accurate timing
	clErr = clFinish(CommandQueue);

	timer.Start();

	//run the kernel N times
	for(unsigned int i = 0; i < NIterations; i++)
	{
		clErr |= clEnqueueNDRangeKernel(CommandQueue, Kernel, Dimensions, NULL, pGlobalWorkSize, pLocalWorkSize, 0, NULL, NULL);
	}
	//wait until the command queue is empty again
	clErr |= clFinish(CommandQueue);

	timer.Stop();

	if(clErr != CL_SUCCESS)
	{
		cout<<"kernel execution failure"<<endl;
		return -1;
	}

	double ms = 1000 * timer.GetElapsedTime() / double(NIterations);

	return ms;
}