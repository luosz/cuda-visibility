#pragma once
#ifndef DEF_H
#define DEF_H

#include <cstdlib>

//float g_SelectedColor[] = { 1.f,1.f,0.f,1.f };
const float D_RGBA[] = { 1.f,1.f,0.f,1.f };
const bool D_APPLY_ALPHA = true;
const bool D_APPLY_COLOR = true;
const int D_WIDTH = 800;
const int D_HEIGHT = 800;
const int D_BIN_COUNT = 256;
const int D_RADIUS = 20;
const bool D_TIME_VARYING_TF_EDITING = true;
const bool D_TIME_VARYING_TF_RESET = true;
const bool D_TIME_VARYING_VWS_OPTIMIZATION = false;
const bool D_TERMPORAL_VISIBILITY = false;
const int D_PATH_MAX = _MAX_PATH;

const int R5 = 5;
const int R9 = 9;

#include <cmath>

#ifndef M_PI
#define M_PI       3.14159265358979323846
#endif

inline float sigma(float radius)
{
	return sqrt(2) / 8 * radius;
}

inline void calc_gaussian_kernel(float kernel[], const int n = 11, const float sigma = 1)
{
	//std::cout << "gaussian_kernel_1d" << std::endl;
	int j = 0;
	for (int x = -(n / 2); x < (n / 2) + 1; x++)
	{
		float val = 1 / (sigma * sqrt(2 * M_PI)) * exp(-x*x / (2. * sigma*sigma));
		kernel[j++] = val;
		//std::cout << val << " ";
	}
	//std::cout << std::endl;
}

#endif // DEF_H
