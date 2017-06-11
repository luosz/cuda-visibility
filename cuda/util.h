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
std::vector<double> gaussian_kernel_1d(int n = 11, double sigma = 1)
{
	std::vector<double> kernel;
	const double pi = M_PI;
	std::cout << "gaussian_kernel_1d" << std::endl;
	for (int x = -(n / 2); x < (n / 2) + 1; x++)
	{
		double val = 1 / (sigma * sqrt(2 * pi)) * exp(-x*x / (2. * sigma*sigma));
		kernel.push_back(val);
		std::cout << val << " ";
	}
	std::cout << std::endl;
	return kernel;
}

void gaussian(int n = 5, double sigma = 1)
{
	int half = n / 2;
	auto kernel = gaussian_kernel_1d(n, sigma);
	std::vector<double> opacity_new = opacity_list;
	for (int i = half; i < opacity_new.size() - half; i++)
	{
		double sum = 0;
		for (int j = 0; j < n; j++)
		{
			int offset = j - half;
			int k = i + offset;
			sum += kernel[j] * get_opacity(k);
		}
		opacity_new[i] = sum;
	}
	opacity_list = opacity_new;

	updateTFWidgetFromOpacityArrays();
	updateOpacityArrayFromTFWidget();
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
