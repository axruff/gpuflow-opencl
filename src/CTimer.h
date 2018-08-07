#ifndef TIMER_H
#define TIMER_H

//Simple wrapper class that can be used to measure time intervals
//using the built-in precision timer of the OS

#ifdef _WIN32

	#include <Windows.h>

#elif defined (__APPLE__) || defined(MACOSX)

	#include <sys/time.h>

#else

	//#include <sys/utime.h>
	#include <sys/time.h>
	#include <time.h>

#endif

class CTimer
{
protected:

#ifdef WIN32
	LARGE_INTEGER		m_StartTime;
	LARGE_INTEGER		m_EndTime;
#else
	struct timeval		m_StartTime;
	struct timeval		m_EndTime;
#endif

public:
	CTimer();

	~CTimer();
	
	void Start();

	void Stop();

	//returns the elapsed time in seconds
	double GetElapsedTime();
};

#endif
