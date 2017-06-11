#pragma once

#ifndef UTIL_H
#define UTIL_H

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
std::vector<float> gaussian_kernel_1d(int n = 11, float sigma = 1)
{
	std::vector<float> kernel;
	std::cout << "gaussian_kernel_1d" << std::endl;
	for (int x = -(n / 2); x < (n / 2) + 1; x++)
	{
		float val = 1 / (sigma * sqrt(2 * M_PI)) * exp(-x*x / (2. * sigma*sigma));
		kernel.push_back(val);
		std::cout << val << " ";
	}
	std::cout << std::endl;
	return kernel;
}

void gaussian(float4 tf[], int count, int kernel_size = 5, float sigma = 1)
{
	auto data = (float4*)malloc(count * sizeof(float4));
	memcpy(data, tf, count * sizeof(float4));
	int half = kernel_size / 2;
	auto kernel = gaussian_kernel_1d(kernel_size, sigma);
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
}

/// Re-maps a number from one range to another.
inline float map_to_range(float val, float src_lower, float src_upper, float target_lower, float target_upper)
{
	val = val < src_lower ? src_lower : val;
	val = val > src_upper ? src_upper : val;
	auto normalised = (val - src_lower) / (src_upper - src_lower);
	return normalised * (target_upper - target_lower) + target_lower;
}

inline float normalise_rgba(int n)
{
	return map_to_range(n, 0, 255, 0, 1);
}

//template<typename T>
//float4 lerp_(T a, T b, float t)
//{
//	return a + t*(b - a);
//}

#endif // UTIL_H
