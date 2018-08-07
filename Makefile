CC 			= g++
CFLAGS 		= -std=c++03 -c -O2 -Wall
LDFLAGS 	= -lOpenCL
SOURCES		= src/Common.cpp src/GPUFullOpticalFlow.cpp src/main.cpp src/CPUOpticalFlow.cpp src/GPUNaiveOpticalFlow.cpp src/OpticalFlowBase.cpp src/CTimer.cpp src/GPUOptimizedOpticalFlow.cpp src/GPUFlowDrivenRobust.cpp src/Image.cpp
OBJECTS 	= $(SOURCES:.cpp=.o)
EXECUTABLE 	= gpuflow

RM 			= rm -f

all: $(SOURCES) $(EXECUTABLE)
	
$(EXECUTABLE): $(OBJECTS) 
	$(CC) $(LDFLAGS) $(OBJECTS) -o $@

.cpp.o:
	$(CC) $(CFLAGS) $< -o $@

clean:
	$(RM) $(OBJECTS) $(EXECUTABLE)