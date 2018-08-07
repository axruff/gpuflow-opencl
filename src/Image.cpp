#include "Image.h"

#include <fstream>
#include <iostream>

#ifndef _WIN32 
	#include <cmath>
	
	#define _ASSERTE(X)
#endif

#define TAG_FLOAT 202021.25

Image::Image() 
	: m_width(0), m_height(0), m_actual_width(0), m_actual_height(0), m_pitch(0), m_bx(0), m_by(0), m_data(NULL)
{

}

Image::Image(int width, int height)
	: m_width(width), m_height(height), m_actual_width(width), m_actual_height(height), m_pitch(0), m_bx(0), m_by(0), m_data(NULL)
{
	allocateDataMemoryWithPadding();
	zeroData();
}

Image::Image(int width, int height, int bx, int by)
	: m_width(width), m_height(height), m_actual_width(width), m_actual_height(height), m_pitch(0), m_bx(bx), m_by(by), m_data(NULL)
{
	allocateDataMemoryWithPadding();
	zeroData();
}

void Image::reinit(int width, int height, int actual_width, int actual_height, int bx, int by)
{
	m_width = width;
	m_height = height;
	m_actual_width = actual_width;
	m_actual_height = actual_height;
	m_bx = bx;
	m_by = by;
	allocateDataMemoryWithPadding();
	zeroData();
}


bool Image::readImagePGM(std::string filename)
{
	const unsigned int size = 256;
	char str[size];
	std::fstream file(filename.c_str(), std::ios::in | std::ios::binary);

	if (!file) {
		std::cout << "Cannot read file: " << filename << std::endl;
		return false;
	}
	file.getline(str, size); // Type: PGM
	file.getline(str, size); // Created by
	while (str[0] == '#') {
		file.getline(str, size);
	}
	// Image width and height
	#ifdef _WIN32   // Windows version
		sscanf_s(str, "%d %d", &m_width, &m_height);
	#else           // Linux version
		sscanf(str, "%d %d", &m_width, &m_height);
	#endif

	m_actual_width = m_width;
	m_actual_height = m_height;

	file.getline(str, size); // Max value

	// Last position
	std::streampos pos = file.tellg();

	allocateDataMemoryWithPadding();

	unsigned char* buf = new unsigned char[m_width];

	file.seekg(pos, std::ios::beg);
	for (int y = 0; y < m_height; y++) {
		file.read(reinterpret_cast<char*>(buf), m_width * sizeof(unsigned char));
		for (int x = 0; x < m_width; x++) {
			m_data[IND(x, y)] = buf[x];
		}
	}

	delete[] buf;

	file.close();
	return true;
}

bool Image::writeImagePGM(std::string filename)
{
	_ASSERTE(m_data != NULL);
	std::ofstream file(filename.c_str(), std::ios_base::binary);
	if (!file.good()) {
		return false;
	}

	file << "P5" << std::endl;
	file << m_width << ' ' << m_height << std::endl;
	file << "255" << std::endl;

	unsigned char* buf = new unsigned char[m_width];

	for (int y = 0; y < m_height; y++) {
		for (int x = 0; x < m_width; x++) {
			unsigned char value = static_cast<unsigned char>( fmax( fmin(m_data[IND(x, y)], 255.), 0.) );
			buf[x] =  value;
		}
		file.write(reinterpret_cast<char*>(buf), m_width * sizeof(unsigned char));
	}

	delete[] buf;

	return true;
}

bool Image::writeImagePGMwithBoundaries(std::string filename)
{
	_ASSERTE(m_data != NULL);
	std::ofstream file(filename.c_str(), std::ios_base::binary);
	if (!file.good()) {
		return false;
	}

	file << "P5" << std::endl;
	file << m_width + 2 * m_bx << ' ' << m_height + 2 * m_by << std::endl;
	file << "255" << std::endl;

	unsigned char* buf = new unsigned char[m_width + 2 * m_by];

	for (int y = 0; y < m_height + 2 * m_by; y++) {
		for (int x = 0; x < m_width + 2 * m_bx; x++) {
			unsigned char value = static_cast<unsigned char>(fmax(fmin(m_data[IND(x - m_bx, y - m_by)], 255.), 0.));
			buf[x] = value;
		}
		file.write(reinterpret_cast<char*>(buf), (m_width + 2 * m_bx) * sizeof(unsigned char));
	}

	delete[] buf;

	return true;
}

bool Image::readMiddlFlowFile(std::string filename, Image& u, Image& v)
{
	const char *dot = strrchr(filename.c_str(), '.');
	if (strcmp(dot, ".flo") != 0 && strcmp(dot, ".um") != 0) {
		std::cout << "ReadFlowFile (" + filename + "): extensions .flo or .um are expected" << std::endl;
		return false;
	}

	FILE *stream = fopen(filename.c_str(), "rb");
	if (stream == 0) {
		std::cout << "ReadFlowFile: could not open " + filename << std::endl;
		return false;
	}

	int width = 0;
	int height = 0;
	float tag = -1;

	if ((int)fread(&tag, sizeof(float), 1, stream) != 1 ||
		(int)fread(&width, sizeof(int), 1, stream) != 1 ||
		(int)fread(&height, sizeof(int), 1, stream) != 1) {
		std::cout << "ReadFlowFile: problem reading file " + filename << std::endl;
		return false;
	}
	if (tag != TAG_FLOAT) { // simple test for correct endian-ness 
		std::cerr << "ReadFlowFile(" + filename + "): wrong tag (possibly due to big-endian machine?)" << std::endl;
		return false;
	}

	// another sanity check to see that integers were read correctly (99999 should do the trick...)
	if (width < 1 || width > 99999) {
		std::cout << "ReadFlowFile(" + filename + "): illegal width " << width << std::endl;
		return false;
	}
	if (height < 1 || height > 99999) {
		std::cout << "ReadFlowFile(" + filename + "): illegal height " << height << std::endl;
		return false;
	}

	u.m_width = width;
	u.m_height = height;
	u.setActualSize(width, height);
	u.allocateDataMemoryWithPadding();
	
	v.m_width = width;
	v.m_height = height;
	v.setActualSize(width, height);
	v.allocateDataMemoryWithPadding();

	float* ptr = new float[2 * width];

	for (int y = 0; y < height; y++) {
		fread(ptr, sizeof(float), 2 * width, stream);

		for (int x = 0; x < width; x++) {
			u.pixel_w(x, y) = (float)ptr[2 * x];
			v.pixel_w(x, y) = (float)ptr[2 * x + 1];
		}
	}

	if (fgetc(stream) != EOF) {
		std::cout << "ReadFlowFile(" + filename + "): file is too long " << height << std::endl;
		return false;
	}

	delete[] ptr;
	fclose(stream);

	return true;
}

void Image::fillBoudaries()
{
	_ASSERTE(m_data != NULL);
	if (m_bx > 0) {
		for (int y = 0; y < m_actual_height; y++) {
			for (int bx = m_bx; bx > 0; bx--) {
				m_data[IND(-bx, y)] = m_data[IND(bx, y)];
				m_data[IND(m_actual_width - 1 + bx, y)] = m_data[IND(m_actual_width - 1 - bx, y)];
			}
		}
	}
	if (m_by > 0) {
		for (int x = 0; x < m_actual_width; x++) {
			for (int by = m_by; by > 0; by--) {
				m_data[IND(x, -by)] = m_data[IND(x, by)];
				m_data[IND(x, m_actual_height - 1 + by)] = m_data[IND(x, m_actual_height - 1 - by)];
			}
		}
	}
}

void Image::resample(const Image& src, Image& dst, float scale)
{
	dst.m_bx = src.m_bx;
	dst.m_by = src.m_by;
	dst.m_width = static_cast<int>(src.m_width * scale);
	dst.m_height = static_cast<int>(src.m_height * scale);
	dst.allocateDataMemoryWithPadding();

	const float tx = static_cast<float>(src.m_width) / dst.m_width;
	const float ty = static_cast<float>(src.m_height) / dst.m_height;
	float C[5] = { 0.f };

	for (int dst_y = 0; dst_y < dst.m_height; dst_y++) {
		for (int dst_x = 0; dst_x < dst.m_width; dst_x++) {
			const int src_x = static_cast<int>(tx * dst_x);
			const int src_y = static_cast<int>(ty * dst_y);
			const float dx = tx * dst_x - src_x;
			const float dy = tx * dst_y - src_y;
			
			for (int jj = 0; jj < 4; jj++) {
				float a0 = src.pixel_v(src_x, src_y - 1 + jj);
				float d0 = src.pixel_v(src_x - 1, src_y - 1 + jj) - a0;
				float d2 = src.pixel_v(src_x + 1, src_y - 1 + jj) - a0;
				float d3 = src.pixel_v(src_x + 2, src_y - 1 + jj) - a0;
				float a1 = -1.f / 3.f * d0 + d2 - 1.f / 6.f * d3;
				float a2 = 1.f / 2.f * d0 + 1.f / 2.f * d2;
				float a3 = -1.f / 6.f * d0 - 1.f / 2.f * d2 + 1.f / 6.f * d3;
				C[jj] = a0 + a1 * dx + a2 * dx * dx + a3 * dx * dx * dx;

				d0 = C[0] - C[1];
				d2 = C[2] - C[1];
				d3 = C[3] - C[1];
				a0 = C[1];
				a1 = -1.f / 3.f * d0 + d2 - 1.f / 6.f * d3;
				a2 = 1.f / 2.f * d0 + 1.f / 2.f * d2;
				a3 = -1.f / 6.f * d0 - 1.f / 2.f * d2 + 1.f / 6.f * d3;

				dst.pixel_w(dst_x, dst_y) = a0 + a1 * dy + a2 * dy * dy + a3 * dy * dy * dy;;
			}
		}
	}
}

void Image::resampleWithoutReallocating(const Image& src, Image& dst, int dst_width, int dst_height)
{
	_ASSERTE(dst.m_width >= dst_width && dst.m_height >= dst_height);
	
	dst.m_actual_width = dst_width;
	dst.m_actual_height = dst_height;

	const float tx = static_cast<float>(src.m_actual_width) / dst_width;
	const float ty = static_cast<float>(src.m_actual_height) / dst_height;
	float C[5] = { 0.f };

	for (int dst_y = 0; dst_y < dst_height; dst_y++) {
		for (int dst_x = 0; dst_x < dst_width; dst_x++) {
			const int src_x = static_cast<int>(tx * dst_x);
			const int src_y = static_cast<int>(ty * dst_y);
			const float dx = tx * dst_x - src_x;
			const float dy = tx * dst_y - src_y;

			for (int jj = 0; jj < 4; jj++) {
				float a0 = src.pixel_v(src_x, src_y - 1 + jj);
				float d0 = src.pixel_v(src_x - 1, src_y - 1 + jj) - a0;
				float d2 = src.pixel_v(src_x + 1, src_y - 1 + jj) - a0;
				float d3 = src.pixel_v(src_x + 2, src_y - 1 + jj) - a0;
				float a1 = -1.f / 3.f * d0 + d2 - 1.f / 6.f * d3;
				float a2 = 1.f / 2.f * d0 + 1.f / 2.f * d2;
				float a3 = -1.f / 6.f * d0 - 1.f / 2.f * d2 + 1.f / 6.f * d3;
				C[jj] = a0 + a1 * dx + a2 * dx * dx + a3 * dx * dx * dx;

				d0 = C[0] - C[1];
				d2 = C[2] - C[1];
				d3 = C[3] - C[1];
				a0 = C[1];
				a1 = -1.f / 3.f * d0 + d2 - 1.f / 6.f * d3;
				a2 = 1.f / 2.f * d0 + 1.f / 2.f * d2;
				a3 = -1.f / 6.f * d0 - 1.f / 2.f * d2 + 1.f / 6.f * d3;

				dst.pixel_w(dst_x, dst_y) = a0 + a1 * dy + a2 * dy * dy + a3 * dy * dy * dy;;
			}
		}
	}
}

/*****************************************************************************/
/*                                                                           */
/*                   Copyright 08/2006 by Dr. Andres Bruhn,                  */
/*     Faculty of Mathematics and Computer Science, Saarland University,     */
/*                           Saarbruecken, Germany.                          */
/*																			 */
/*              MODIFIED by Alexey Ershov	<ershov.alexey@gmail.com>		 */
/*****************************************************************************/

void resample_1d
(
/*************************************************************/
float*	u,	/* in     : input vector, size 1..n                          */
int		n,	/* in     : size of input vector                             */
int		m,	/* in     : size of output vector                            */
float*	v	/* out    : output vector, size 1..m                         */
/*************************************************************/
)
/* Area-based resampling: Transforms a 1D image u of size n into an image v  */
/* of size m by integration over piecewise constant functions. Conservative. */
{
	/****************************************************/
	int     i, k;            /* loop variables                                   */
	float  hu, hv;          /* grid sizes                                       */
	float  uleft, uright;   /* boundaries                                       */
	float  vleft, vright;   /* boundaries                                       */
	float  fac;             /* normalization factor                             */
	/****************************************************/

	/*****************************************************************************/
	/* (1/2) Special cases of area-based resampling are computed efficiently     */
	/*****************************************************************************/

	/* fast interpolation for output images of even size */
	if (m == 2 * n)
	{
		/* one cell is devided in two cells with equal value */
		for (i = 1; i <= n; i++)
		{
			v[i * 2 - 1] = u[i];
			v[i * 2] = u[i];
		}
		return;
	}

	/* fast restriction for input images of even size */
	if (2 * m == n)
	{
		/* two celss are melted to a larger cell with averaged value */
		for (i = 1; i <= m; i++)
		{
			v[i] = 0.5f*(u[i * 2 - 1] + u[i * 2]);
		}
		return;
	}

	/*****************************************************************************/
	/* (2/2) Remaining cases require more complex algorithm                      */
	/*****************************************************************************/

	/* initializations */
	/*************************************************/
	hu = 1.0f / (float)n;     /* grid size of u                                */
	hv = 1.0f / (float)m;     /* grid size of v                                */
	uleft = 0.0f;             /* left interval boundary of u                   */
	vleft = 0.0f;             /* left interval boundary of v                   */
	k = 1;					  /* index for u                                   */
	fac = hu / hv;            /* for normalization                             */
	/*************************************************/

	/*---- loop ----*/
	for (i = 1; i <= m; i++)
		/* calculate v[i] by integrating the piecewise constant function u */
	{
		/* calculate right interval boundaries */
		uright = uleft + hu;
		vright = vleft + hv;

		if (uright > vright)
			/* since uleft <= vleft, the entire v-cell i is in the u-cell k */
			v[i] = u[k];
		else
		{
			/* consider fraction alpha of the u-cell k in v-cell i */
			v[i] = (uright - vleft) * n * u[k++];

			/* update */
			uright = uright + hu;

			/* consider entire u-cells inside v-cell i */
			while (uright <= vright)
				/* u-cell k lies entirely in v-cell i; sum up */
			{
				v[i] = v[i] + u[k++];
				uright = uright + hu;
			}

			/* consider fraction beta of the u-cell k in v-cell i */
			v[i] = v[i] + (1.0f - (uright - vright) * n) * u[k];

			/* normalization */
			v[i] = v[i] * fac;
		} /* else */

		/* update */
		uleft = uright - hu;
		vleft = vright;
		/* now it holds: uleft <= vleft */
	}  /* for i */
	return;
}

void resample_2d_x
(
/*************************************************************/
const Image& src,	/* in   : input image					 */
	  Image& dst	/* out  : output image					 */
/*************************************************************/
)
/* resample a 2-D image in x-direction using area-based resampling */
{
	/****************************************************/
	int    x, y;             /* loop variables                                   */
	float *uhelp, *vhelp;    /* auxiliary vectors                                */
	/****************************************************/

	/* allocate memory */
	uhelp = new float[src.actual_width() + 2];
	vhelp = new float[dst.actual_width() + 2];

	/* resample image linewise in x-direction */
	for (y = 0; y < src.actual_height(); y++)
	{
		/* initialise left boundary of 1-D array with zero */
		uhelp[0] = src.pixel_r(0, y);

		/* copy current line in this 1-D array */
		for (x = 0; x < src.actual_width(); x++)
			uhelp[x + 1] = src.pixel_r(x, y);

		/* initialise right boundary of this 1-D array with zero */
		uhelp[src.actual_width() + 1] = src.pixel_r(src.actual_width() - 1, y);

		/* resample this 1-D array */
		resample_1d(uhelp, src.actual_width(), dst.actual_width(), vhelp);

		/* copy resmapled array in corresponding output line */
		for (x = 0; x < dst.actual_width(); x++)
			dst.pixel_r(x, y) = vhelp[x + 1];
	}

	/* free memory */
	delete[] uhelp;
	delete[] vhelp;

	return;
}

void resample_2d_y
(
/*************************************************************/
const Image& src,	/* in   : input image					 */
	  Image& dst	/* out  : output image					 */
/*************************************************************/
)
/* resample a 2-D image in y-direction using area-based resampling */
{
	/****************************************************/
	int    x, y;             /* loop variables                                   */
	float *uhelp, *vhelp;    /* auxiliary vectors                                */
	/****************************************************/

	/* allocate memory */
	uhelp = new float[src.actual_height() + 2];
	vhelp = new float[dst.actual_height() + 2];

	/* resample image columnwise in y-direction */
	for (x = 0; x < src.actual_width(); x++)
	{
		/* initialsie left boundary of 1-D array with zero */
		uhelp[0] = src.pixel_r(x, 0);

		/* copy current column in this 1-D array */
		for (y = 0; y < src.actual_height(); y++)
			uhelp[y + 1] = src.pixel_r(x, y);

		/* initialise right boundary of this 1-D array with zero */
		uhelp[src.actual_height() + 1] = src.pixel_r(x, src.actual_height() - 1);

		/* resample this 1-D array */
		resample_1d(uhelp, src.actual_height(), dst.actual_height(), vhelp);

		/* copy resmapled array in corresponding output column */
		for (y = 0; y < dst.actual_height(); y++)
			dst.pixel_w(x, y) = vhelp[y + 1];
	}

	/* free memory */
	delete[] uhelp;
	delete[] vhelp;

	return;
}

void Image::resampleAreaBasedWithoutReallocating(const Image& src, Image& dst, int dst_width, int dst_height)
{
	_ASSERTE(dst.m_width >= dst_width && dst.m_height >= dst_height);

	dst.m_actual_width = dst_width;
	dst.m_actual_height = dst_height;

	/* if interpolation */
	if (dst.actual_height() >= src.actual_height()) {
		Image tmp(dst.actual_width(), src.actual_height());
		resample_2d_x(src, tmp);
		resample_2d_y(tmp, dst);
	} 
	/* if restriction */
	else {
		Image tmp(src.actual_width(), dst.actual_height());
		resample_2d_y(src, tmp);
		resample_2d_x(tmp, dst);
	}
}

void Image::backwardRegistration(const Image& src1, // in	: 1st image
								 const Image& src2, // in	: 2nd image
									   Image& dst2,	// out	: 2nd image (motion compensated)
								 const Image& u,	// in	: x-component of displacement field
								 const Image& v,	// in	: y-component of displacement field
								 float hx,			// in	: grid spacing in x-direction
								 float hy			// in	: grid spacing in y-direction
	)
{
	int   y, x;                 // loop variables                                 
	int   yy, xx;               // pixel coordinates                              
	float yy_fp, xx_fp;         // subpixel coordinates                           
	float delta_y, delta_x;     // subpixel displacement                          

	float hx_1 = 1.f / hx;
	float hy_1 = 1.f / hy;

	dst2.m_actual_width = src2.m_actual_width;
	dst2.m_actual_height = src2.m_actual_height;

	for (y = 0; y < src2.m_actual_height; y++) {
		for (x = 0; x < src2.m_actual_width; x++) {

			// Compute subpixel location 
			yy_fp = y + (v.pixel_v(x, y) * hy_1);
			xx_fp = x + (u.pixel_v(x, y) * hx_1);

			// If the required image information is out of bounds 
			if ((yy_fp < 0) || (xx_fp < 0) || 
				(yy_fp > (src2.m_actual_height - 1)) || 
				(xx_fp > (src2.m_actual_width - 1)) 
				//|| Toolbox<float >::isNAN(yy_fp) || Toolbox<float >::isNAN(xx_fp)
				){
				// assume zero flow, i.e. set warped 2nd image to 1st image 
				dst2.pixel_w(x, y) = src1.pixel_v(x, y);
			}
			// If required image information is available 
			else {
				// compute integer index of upper left pixel 
				yy = static_cast<int>(std::floor(yy_fp));
				xx = static_cast<int>(std::floor(xx_fp));

				// compute subpixel displacement 
				delta_y = yy_fp - static_cast<float>(yy);
				delta_x = xx_fp - static_cast<float>(xx);

				// perform bilinear interpolation 
				dst2.pixel_w(x, y) = (1.f - delta_y) * (1.f - delta_x)	* src2.pixel_v(xx, yy)
								 + (1.f - delta_y) * delta_x			* src2.pixel_v(xx + 1, yy)
								 + delta_y		   * (1.f - delta_x)	* src2.pixel_v(xx, yy + 1)
								 + delta_y		   * delta_x			* src2.pixel_v(xx + 1, yy + 1);
			}
		}
	}
}

Image& Image::operator+= (const Image& image) 
{
	_ASSERTE(this->m_actual_width == image.m_actual_width && this->m_actual_height == image.m_actual_height);

	for (int y = 0; y < this->m_actual_height; ++y) {
		for (int x = 0; x < this->m_actual_width; ++x) {
			this->pixel_w(x, y) += image.pixel_r(x, y);
		}
	}
	return *this;
}

Image& Image::operator= (const Image& image)
{
	_ASSERTE(this->m_width >= image.m_actual_width && this->m_height >= image.m_actual_height);
	this->m_actual_width = image.m_actual_width;
	this->m_actual_height = image.m_actual_height;

	for (int y = 0; y < this->m_actual_height; ++y) {
		for (int x = 0; x < this->m_actual_width; ++x) {
			this->pixel_w(x, y) = image.pixel_r(x, y);
		}
	}
	return *this;
}

/*****************************************************************************/
/*                                                                           */
/*                   Copyright 08/2006 by Dr. Andres Bruhn                   */
/*     Faculty of Mathematics and Computer Science, Saarland University,     */
/*                           Saarbruecken, Germany.                          */
/*																			 */
/*						   Modified by Alexey Ershov		                 */
/*																			 */
/*****************************************************************************/

struct RGBColor
{
	int r;
	int g;
	int b;

	RGBColor();

	RGBColor(int r, int g, int b);
};

RGBColor::RGBColor()
	: r(0), g(0), b(0) {}

RGBColor::RGBColor(int r, int g, int b)
	: r(r), g(g), b(b) {}

int ConvertToByte(int num)
{
	return (num >= 255) * 255 + ((num < 255) && (num > 0))* num;
}

RGBColor ConvertToRGB(float x, float y)
{
	/********************************************************/
	float Pi;          /* pi                                                   */
	float amp;         /* amplitude (magnitude)                                */
	float phi;         /* phase (angle)                                        */
	float alpha, beta; /* weights for linear interpolation                     */
	/********************************************************/

	RGBColor rgb;

	/* unknown flow */
	if ((std::fabs(x) >  1e6) || (std::fabs(y) >  1e6) || x != x || y != y ) {
		x = 0.0;
		y = 0.0;
	}

	/* set pi */
	Pi = 2.f * acos(0.f);

	/* determine amplitude and phase (cut amp at 1) */
	amp = sqrt(x * x + y * y);
	if (amp > 1) amp = 1;
	if (x == 0.f)
		if (y >= 0.f) phi = 0.5f * Pi;
		else phi = 1.5f * Pi;
	else if (x > 0.f)
		if (y >= 0.f) phi = atan(y / x);
		else phi = 2.f * Pi + atan(y / x);
	else phi = Pi + atan(y / x);

	phi = phi / 2.f;

	// interpolation between red (0) and blue (0.25 * Pi)
	if ((phi >= 0.f) && (phi < 0.125f * Pi)) {
		beta = phi / (0.125f * Pi);
		alpha = 1.f - beta;
		rgb.r = (int)floor(amp * (alpha * 255.0f + beta * 255.0f));
		rgb.g = (int)floor(amp * (alpha *   0.0f + beta *   0.0f));
		rgb.b = (int)floor(amp * (alpha *   0.0f + beta * 255.0f));
	}
	if ((phi >= 0.125f * Pi) && (phi < 0.25f * Pi)) {
		beta = (phi - 0.125f * Pi) / (0.125f * Pi);
		alpha = 1.0f - beta;
		rgb.r = (int)floor(amp * (alpha * 255.0f + beta *  64.0f));
		rgb.g = (int)floor(amp * (alpha *   0.0f + beta *  64.0f));
		rgb.b = (int)floor(amp * (alpha * 255.0f + beta * 255.0f));
	}
	// interpolation between blue (0.25 * Pi) and green (0.5 * Pi)
	if ((phi >= 0.25f * Pi) && (phi < 0.375f * Pi)) {
		beta = (phi - 0.25f * Pi) / (0.125f * Pi);
		alpha = 1.0f - beta;
		rgb.r = (int)floor(amp * (alpha *  64.0f + beta *   0.0f));
		rgb.g = (int)floor(amp * (alpha *  64.0f + beta * 255.0f));
		rgb.b = (int)floor(amp * (alpha * 255.0f + beta * 255.0f));
	}
	if ((phi >= 0.375f * Pi) && (phi < 0.5f * Pi)) {
		beta = (phi - 0.375f * Pi) / (0.125f * Pi);
		alpha = 1.0f - beta;
		rgb.r = (int)floor(amp * (alpha *   0.0f + beta *   0.0f));
		rgb.g = (int)floor(amp * (alpha * 255.0f + beta * 255.0f));
		rgb.b = (int)floor(amp * (alpha * 255.0f + beta *   0.0f));
	}
	// interpolation between green (0.5 * Pi) and yellow (0.75 * Pi)
	if ((phi >= 0.5f * Pi) && (phi < 0.75f * Pi)) {
		beta = (phi - 0.5f * Pi) / (0.25f * Pi);
		alpha = 1.0f - beta;
		rgb.r = (int)floor(amp * (alpha * 0.0f + beta * 255.0f));
		rgb.g = (int)floor(amp * (alpha * 255.0f + beta * 255.0f));
		rgb.b = (int)floor(amp * (alpha * 0.0f + beta * 0.0f));
	}
	// interpolation between yellow (0.75 * Pi) and red (Pi)
	if ((phi >= 0.75f * Pi) && (phi <= Pi)) {
		beta = (phi - 0.75f * Pi) / (0.25f * Pi);
		alpha = 1.0f - beta;
		rgb.r = (int)floor(amp * (alpha * 255.0f + beta * 255.0f));
		rgb.g = (int)floor(amp * (alpha * 255.0f + beta * 0.0f));
		rgb.b = (int)floor(amp * (alpha * 0.0f + beta * 0.0f));
	}

	/* check RGBColor range */
	rgb.r = ConvertToByte(rgb.r);
	rgb.g = ConvertToByte(rgb.g);
	rgb.b = ConvertToByte(rgb.b);

	return rgb;
}

void Image::saveOpticalFlowRGB(const Image& u, const Image& v, float flow_scale, std::string filename)
{
	std::ofstream file(filename.c_str(), std::ios_base::binary);
	if (!file.good()) {
		return;
	}

	file << "P6" << std::endl;
	file << u.m_actual_width << ' ' << u.m_actual_height << std::endl;
	file << "255" << std::endl;

	float factor = 1.f / flow_scale;

	for (int y = 0; y < u.m_actual_height; ++y) {
		for (int x = 0; x < u.m_actual_width; ++x) {
			
			RGBColor rgb = ConvertToRGB(u.pixel_r(x, y) * factor, v.pixel_r(x, y) * factor);

			unsigned char R = rgb.r;
			unsigned char G = rgb.g;
			unsigned char B = rgb.b;

			file.write(reinterpret_cast<char*>(&R), sizeof(unsigned char));
			file.write(reinterpret_cast<char*>(&G), sizeof(unsigned char));
			file.write(reinterpret_cast<char*>(&B), sizeof(unsigned char));
		}
	}
}

void Image::allocateDataMemoryWithPadding()
{
	if (m_width == 0 || m_height == 0) {
		return;
	}

	if (m_data) {
		delete[] m_data;
	}
	// Fullwidth with boundary pixels
	int fullWidth = m_width + m_bx * 2;
	m_pitch = (fullWidth % 32 == 0) ? fullWidth : fullWidth + 32 - (fullWidth % 32);
	m_data = new float[m_pitch * (m_height  + 2 * m_by)];
}

Image::~Image()
{
	if (m_data) {
		delete[] m_data;
	}
}
