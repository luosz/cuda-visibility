#pragma once

#ifndef UTIL_H
#define UTIL_H

#include "def.h"

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


//// A recursive binary search function. It returns location of x in
//// given array arr[l..r] is present, otherwise -1
//int binary_search(float arr[], int l, int r, float x)
//{
//	float epsilon = std::numeric_limits<float>::epsilon();
//	if (r >= l)
//	{
//		int mid = l + (r - l) / 2;
//
//		// If the element is present at the middle itself
//		if (abs(arr[mid] - x) < epsilon)
//		{
//			return mid;
//		}
//
//		// If element is smaller than mid, then it can only be present
//		// in left subarray
//		if (x < arr[mid])
//		{
//			return binary_search(arr, l, mid - 1, x);
//		}
//
//		// Else the element can only be present in right subarray
//		return binary_search(arr, mid + 1, r, x);
//	}
//
//	// We reach here when element is not present in array
//	return -1;
//}

int linear_search(float arr[], int first, int last, float x)
{
	int ans = -1;
	for (int i = first; i < last; i++)
	{
		if (arr[i] < x)
		{
			ans = i;
		}
	}
	return ans;
}

int binary_search(float arr[], int first, int last, float x)
{
	if (x < arr[first])
	{
		return -1;
	}
	while (first < last)
	{
		int mid = first + (last - first) / 2;
		if (mid == first)
		{
			break;
		}
		if (x < arr[mid])
		{
			last = mid;
		}
		else
		{
			first = mid;
		}
	}
	return first;
}

#endif // UTIL_H
