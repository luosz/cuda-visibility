#pragma once

#ifndef GAUSSIAN_H
#define GAUSSIAN_H

#include <stdio.h>      /* printf, scanf, NULL */
#include <stdlib.h>     /* malloc, free, rand */
#include <helper_cuda.h>
#include <helper_math.h>

#ifndef M_PI
#define M_PI       3.14159265358979323846
#endif

/**
http://stackoverflow.com/questions/11209115/creating-gaussian-filter-of-required-length-in-python
from math import pi, sqrt, exp
def gauss(n=11,sigma=1):
r = range(-int(n/2),int(n/2)+1)
return [1 / (sigma * sqrt(2*pi)) * exp(-float(x)**2/(2*sigma**2)) for x in r]
*/

void gaussian_kernel_1d(float kernel[], const int n = 11, const float sigma = 1)
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

void gaussian(float4 tf[], const int count, const int kernel_size = 5, const float sigma = 1)
{
	int half = kernel_size / 2;
	auto kernel = (float*)malloc(kernel_size * sizeof(float));
	gaussian_kernel_1d(kernel, kernel_size, sigma);
	auto data = (float4*)malloc(count * sizeof(float4));
	memcpy(data, tf, count * sizeof(float4));
	for (int i = half; i < count - half; i++)
	{
		float4 sum = make_float4(0, 0, 0, 0);
		for (int j = 0; j < kernel_size; j++)
		{
			int k = i + j - half;
			sum += kernel[j] * tf[k];
		}
		data[i] = sum;
	}
	memcpy(tf, data, count * sizeof(float4));
	free(data);
	free(kernel);
}

#endif // GAUSSIAN_H
