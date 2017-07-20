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

#endif // UTIL_H
