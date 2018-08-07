#pragma once

#include <string>
// Linux declaration
#ifndef _WIN32 
	#include <cstring>
#endif

#define IND(X, Y) (((Y) + m_by) * m_pitch + ((X) + m_bx))

class Image
{
private:
	int m_width;		// Image width
	int m_height;		// Image height
	int m_actual_width;	// Actual image width
	int m_actual_height;// Actual image height
	int m_pitch;		// We expand image width if it isn't a multiple of 32 to provide coalesced data accesses
	int m_bx;			// Boudary size X
	int m_by;			// Boudary size Y

	float* m_data;	// Image data

public:
	Image();
	Image(int width, int height);
	Image(int width, int height, int bx, int by);
	~Image();

	/* returns reference to pixel in data array w - write / r - read */
	inline float& pixel_w(int x, int y) { return m_data[IND(x, y)]; };
	inline float& pixel_r(int x, int y) const { return m_data[IND(x, y)]; };
	/* returns pixel value from (x, y) position if x and y are valid, 0.f otherwise	*/
	inline float  pixel_v(int x, int y) const { return (x < 0 || x >= m_actual_width || y < 0 || y >= m_actual_height) ? 0.f : m_data[IND(x, y)]; };
	
	inline int width() const { return m_width; };
	inline int height() const { return m_height; };
	inline int pitch() const { return m_pitch; };
	inline int actual_width() const { return m_actual_width; };
	inline int actual_height() const { return m_actual_height; };
	inline float* data_ptr() { return m_data; };
	void swap_data(Image& swap) { std::swap(this->m_data, swap.m_data); };

	void reinit(int width, int height, int actual_width, int actual_height, int bx, int by);
	void setActualWidth(int width) { m_actual_width = width; };
	void setActualHeight(int height) { m_actual_height = height; };
	void setActualSize(int width, int height) { m_actual_width = width; m_actual_height = height; };
	
	/* fills boudaries with mirrored image */
	void fillBoudaries();
	void zeroData() { if (m_data) std::memset(m_data, 0, (m_height + 2 * m_by) * m_pitch * sizeof(float)); };

	bool readImagePGM(std::string filename);
	bool writeImagePGM(std::string filename);
	bool writeImagePGMwithBoundaries(std::string filename);
	static bool readMiddlFlowFile(std::string filename, Image& u, Image& v);

	static void resample(const Image& src, Image& dst, float scale);
	static void resampleWithoutReallocating(const Image& src, Image& dst, int dst_width, int dst_height);

	static void resampleAreaBasedWithoutReallocating(const Image& src, Image& dst, int dst_width, int dst_height);

	static void backwardRegistration(const Image& src1, const Image& src2, Image& dst2, const Image& u, const Image& v, float hx, float hy);

	static void saveOpticalFlowRGB(const Image& u, const Image& v, float flow_scale, std::string filename);

	Image& operator+= (const Image& image);
	Image& operator= (const Image& image);

private:
	void allocateDataMemoryWithPadding();
};

