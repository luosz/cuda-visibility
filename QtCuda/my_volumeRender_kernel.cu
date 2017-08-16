/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

 // Simple 3D volume renderer

#ifndef _VOLUMERENDER_KERNEL_CU_
#define _VOLUMERENDER_KERNEL_CU_

#include <helper_cuda.h>
#include <helper_math.h>
#include <iostream>
#include <stdio.h>
#include "define.cuh"
using namespace std;

typedef unsigned int  uint;
typedef unsigned char uchar;

cudaArray *d_volumeArray = 0;
cudaArray *d_transferFuncArray;

typedef unsigned char VolumeType;
//typedef unsigned short VolumeType;

texture<VolumeType, 3, cudaReadModeNormalizedFloat> tex;         // 3D texture
texture<float4, 1, cudaReadModeElementType>         transferTex; // 1D transfer function texture

texture<float, cudaTextureType3D, cudaReadModeElementType>  volumeTexIn;
surface<void, 3>                                    volumeTexOut;
cudaArray *d_visibilityArray = 0;

__device__ __managed__ VolumeType *raw_volume = NULL;
__device__ __managed__ char *volume_file = NULL;
__device__ __managed__ float *visVolume = NULL;
__device__ __managed__ int *countVolume = NULL;
__device__ __managed__ float *depthVolume = NULL;
__device__ __managed__ cudaExtent sizeOfVolume;// = make_cudaExtent(32, 32, 32);
typedef float VisibilityType;
texture<VisibilityType, 3, cudaReadModeElementType> visTex;         // 3D texture
texture<VolumeType, 3, cudaReadModeElementType> volTex;         // 3D texture
//texture<VisibilityType, 3, cudaReadModeNormalizedFloat> visTex;         // 3D texture

const int BIN_COUNT = D_BIN_COUNT;
__device__ __managed__ float histogram[BIN_COUNT] = {0};
__device__ __managed__ float histogram2[BIN_COUNT] = { 0 };
__device__ __managed__ float histogram3[BIN_COUNT] = { 0 };
__device__ __managed__ float histogram4[BIN_COUNT] = { 0 };
__device__ __managed__ float4 tf_array[BIN_COUNT] = { 0 };
__device__ __managed__ float4 tf_array0[BIN_COUNT] = { 0 };
__device__ __managed__ int radius = D_RADIUS;
__device__ __managed__ float g5[R5*R5*R5] = { 0 };
__device__ __managed__ float g9[R9*R9*R9] = { 0 };
__device__ __managed__ float *saliencyVolume = NULL;
__device__ __managed__ float *vwsVolume = NULL;
bool saliency_once = false;
__device__ __managed__ unsigned char *featureVolume = NULL;
__device__ __managed__ int feature_number = 0;
__device__ __managed__ float feature_array[BIN_COUNT] = { 0 };
__device__ __managed__ float feature_vws_array[BIN_COUNT] = { 0 };

// GUI settings
//float g_SelectedColor[] = { 1.f,1.f,0.f,1.f };
float g_SelectedColor[] = { D_RGBA[0], D_RGBA[1], D_RGBA[2], D_RGBA[3] };
bool g_ApplyAlpha = D_APPLY_ALPHA;
bool g_ApplyColor = D_APPLY_COLOR;

// apply, save and discard operations
bool apply_blend = false;
bool discard_table = false;
bool save_histogram = false;
bool gaussian_histogram = false;
bool backup_table = false;

extern "C" float4 rgb_to_lch(float4 rgba);
extern "C" int iDivUp(int a, int b);

extern "C" VolumeType * get_raw_volume()
{
	return raw_volume;
}

extern "C" unsigned char * get_feature_volume()
{
	return featureVolume;
}

extern "C" float * get_vws_volume()
{
	return vwsVolume;
}

extern "C" int get_feature_number()
{
	return feature_number;
}

extern "C" void set_feature_number(int val)
{
	feature_number = val;
}

extern "C" float * get_feature_array()
{
	return feature_array;
}

extern "C" float * get_feature_vws_array()
{
	return feature_vws_array;
}

typedef float(*Pointer)[4];
extern "C" Pointer get_SelectedColor()
{
	return &g_SelectedColor;
}

extern "C" void set_SelectedColor(float r, float g, float b)
{
	g_SelectedColor[0] = r;
	g_SelectedColor[1] = g;
	g_SelectedColor[2] = b;
}

extern "C" bool* get_ApplyColor()
{
	return &g_ApplyColor;
}

extern "C" bool* get_ApplyAlpha()
{
	return &g_ApplyAlpha;
}

extern "C" int get_region_size()
{
	return radius;
}

extern "C" float4* get_tf_array()
{
	return tf_array;
}

extern "C" void backup_tf()
{
	memcpy(tf_array0, tf_array, sizeof(tf_array));
}

extern "C" void bind_tf_texture()
{
	cudaChannelFormatDesc channelDesc2 = cudaCreateChannelDesc<float4>();
	cudaArray *d_transferFuncArray;
	checkCudaErrors(cudaMallocArray(&d_transferFuncArray, &channelDesc2, sizeof(tf_array) / sizeof(float4), 1));
	checkCudaErrors(cudaMemcpyToArray(d_transferFuncArray, 0, 0, tf_array, sizeof(tf_array), cudaMemcpyHostToDevice));
	// Bind the array to the texture
	checkCudaErrors(cudaBindTextureToArray(transferTex, d_transferFuncArray, channelDesc2));
}

extern "C" void restore_tf()
{
	memcpy(tf_array, tf_array0, sizeof(tf_array));
	bind_tf_texture();
}

extern "C" int get_bin_count()
{
	return BIN_COUNT;
}

extern "C" bool get_save()
{
	return save_histogram;
}

extern "C" void set_save(bool value)
{
	save_histogram = value;
	printf("set save %s\n", save_histogram ?"true":"false");
}

extern "C" bool get_apply()
{
	return apply_blend;
}

extern "C" void set_apply(bool value)
{
	apply_blend = value;
	printf("set apply %s\n", apply_blend ? "true" : "false");
}

extern "C" bool get_discard()
{
	return discard_table;
}

extern "C" void set_discard(bool value)
{
	discard_table = value;
	printf("set discard %s\n", discard_table ? "true" : "false");
}

extern "C" bool get_gaussian()
{
	return gaussian_histogram;
}

extern "C" void set_gaussian(bool value)
{
	gaussian_histogram = value;
	printf("set gaussian %s\n", gaussian_histogram ? "true" : "false");
}

extern "C" bool get_backup()
{
	return backup_table;
}

extern "C" void set_backup(bool value)
{
	backup_table = value;
	printf("set backup %s\n", backup_table ? "true" : "false");
}

extern "C" void set_volume_file(const char *file, int n)
{
	n = n + 1;
	if (!volume_file)
	{
		checkCudaErrors(cudaMallocManaged(&volume_file, sizeof(float) * n));
	}
	memcpy(volume_file, file, n);
}

extern "C" void blend_tf(float3 color)
{
	float hist[BIN_COUNT];
	float max = 0;
	for (int i = 0; i < BIN_COUNT; i++)
	{
		if (max < histogram2[i])
		{
			max = histogram2[i];
		}
	}

	for (int i = 0; i < BIN_COUNT; i++)
	{
		hist[i] = histogram2[i] / max;
	}
	for (int i = 0; i < BIN_COUNT; i++)
	{
		auto c = make_float3(tf_array[i].x, tf_array[i].y, tf_array[i].z);
		auto c2 = lerp(c, color, hist[i]);
		if (hist[i] > 0.5)
		{
			printf("%g r %g %g g %g %g b %g %g \n", i/(float)BIN_COUNT, tf_array[i].x, c2.x, tf_array[i].y, c2.y, tf_array[i].z, c2.z);
		}
		tf_array[i].x = c2.x;
		tf_array[i].y = c2.y;
		tf_array[i].z = c2.z;
	}

	bind_tf_texture();
}

extern "C" void blend_tf_rgba(float3 color)
{
	float hist[BIN_COUNT], hist2[BIN_COUNT];
	float sum = 0;
	for (int i = 0; i < BIN_COUNT; i++)
	{
		sum += histogram[i];
	}
	for (int i = 0; i < BIN_COUNT; i++)
	{
		hist[i] = histogram[i] / sum;
	}
	float sum2 = 0;
	for (int i = 0; i < BIN_COUNT; i++)
	{
		sum2 += histogram2[i];
	}
	for (int i = 0; i < BIN_COUNT; i++)
	{
		hist2[i] = histogram2[i] / sum2;
	}
	float max = 0;
	for (int i = 0; i < BIN_COUNT; i++)
	{
		histogram3[i] = hist2[i] - hist[i];
		auto m = fabsf(histogram3[i]);
		max = max < m ? m : max;
	}

	// apply Gaussian filter to relateive visibility histogram
	memcpy(histogram4, histogram3, BIN_COUNT * sizeof(float));
	gaussian(histogram4, BIN_COUNT);

	// normalize histogram3
	for (int i = 0; i < BIN_COUNT; i++)
	{
		histogram3[i] /= max;
	}

	// normalize histogram4
	max = 0;
	for (int i = 0; i < BIN_COUNT; i++)
	{
		auto m = fabsf(histogram4[i]);
		max = max < m ? m : max;
	}
	for (int i = 0; i < BIN_COUNT; i++)
	{
		histogram4[i] /= max;
	}

	if (g_ApplyColor)
	{
		for (int i = 0; i < BIN_COUNT; i++)
		{
			auto c = make_float3(tf_array[i].x, tf_array[i].y, tf_array[i].z);
			auto t = histogram3[i] > 0 ? histogram3[i] : 0;
			auto c2 = lerp(c, color, t);
			tf_array[i].x = c2.x;
			tf_array[i].y = c2.y;
			tf_array[i].z = c2.z;
		}
	}

	if (g_ApplyAlpha)
	{
		for (int i = 0; i < BIN_COUNT; i++)
		{
			tf_array[i].w = lerp(tf_array[i].w, histogram3[i] > 0 ? 1 : 0, fabsf(histogram3[i]));
		}
	}

	bind_tf_texture();
}

extern "C" void gaussian_tf(float3 color)
{
	float hist[BIN_COUNT], hist2[BIN_COUNT];
	float sum = 0;
	for (int i = 0; i < BIN_COUNT; i++)
	{
		sum += histogram[i];
	}
	for (int i = 0; i < BIN_COUNT; i++)
	{
		hist[i] = histogram[i] / sum;
	}
	float sum2 = 0;
	for (int i = 0; i < BIN_COUNT; i++)
	{
		sum2 += histogram2[i];
	}
	for (int i = 0; i < BIN_COUNT; i++)
	{
		hist2[i] = histogram2[i] / sum2;
	}
	float max = 0;
	for (int i = 0; i < BIN_COUNT; i++)
	{
		histogram3[i] = hist2[i] - hist[i];
		auto m = fabsf(histogram3[i]);
		max = max < m ? m : max;
	}

	// apply Gaussian filter to relateive visibility histogram
	memcpy(histogram4, histogram3, BIN_COUNT * sizeof(float));
	gaussian(histogram4, BIN_COUNT);

	// normalize histogram3
	for (int i = 0; i < BIN_COUNT; i++)
	{
		histogram3[i] /= max;
	}

	// normalize histogram4
	max = 0;
	for (int i = 0; i < BIN_COUNT; i++)
	{
		auto m = fabsf(histogram4[i]);
		max = max < m ? m : max;
	}
	for (int i = 0; i < BIN_COUNT; i++)
	{
		histogram4[i] /= max;
	}

	if (g_ApplyColor)
	{
		for (int i = 0; i < BIN_COUNT; i++)
		{
			auto c = make_float3(tf_array[i].x, tf_array[i].y, tf_array[i].z);
			auto t = histogram4[i] > 0 ? histogram4[i] : 0;
			auto c2 = lerp(c, color, t);
			tf_array[i].x = c2.x;
			tf_array[i].y = c2.y;
			tf_array[i].z = c2.z;
		}
	}

	if (g_ApplyAlpha)
	{
		for (int i = 0; i < BIN_COUNT; i++)
		{
			auto a = histogram4[i] > 0 ? 1 : 0;
			auto t = fabsf(histogram4[i]);
			tf_array[i].w = lerp(tf_array[i].w, a, t);
		}
	}

	bind_tf_texture();
}

typedef struct
{
    float4 m[3];
} float3x4;

__constant__ float3x4 c_invViewMatrix;  // inverse view matrix

struct Ray
{
    float3 o;   // origin
    float3 d;   // direction
};

// intersect ray with a box
// http://www.siggraph.org/education/materials/HyperGraph/raytrace/rtinter3.htm

__device__
int intersectBox(Ray r, float3 boxmin, float3 boxmax, float *tnear, float *tfar)
{
    // compute intersection of ray with all six bbox planes
    float3 invR = make_float3(1.0f) / r.d;
    float3 tbot = invR * (boxmin - r.o);
    float3 ttop = invR * (boxmax - r.o);

    // re-order intersections to find smallest and largest on each axis
    float3 tmin = fminf(ttop, tbot);
    float3 tmax = fmaxf(ttop, tbot);

    // find the largest tmin and the smallest tmax
    float largest_tmin = fmaxf(fmaxf(tmin.x, tmin.y), fmaxf(tmin.x, tmin.z));
    float smallest_tmax = fminf(fminf(tmax.x, tmax.y), fminf(tmax.x, tmax.z));

    *tnear = largest_tmin;
    *tfar = smallest_tmax;

    return smallest_tmax > largest_tmin;
}

// transform vector by matrix (no translation)
__device__
float3 mul(const float3x4 &M, const float3 &v)
{
    float3 r;
    r.x = dot(v, make_float3(M.m[0]));
    r.y = dot(v, make_float3(M.m[1]));
    r.z = dot(v, make_float3(M.m[2]));
    return r;
}

// transform vector by matrix with translation
__device__
float4 mul(const float3x4 &M, const float4 &v)
{
    float4 r;
    r.x = dot(v, M.m[0]);
    r.y = dot(v, M.m[1]);
    r.z = dot(v, M.m[2]);
    r.w = 1.0f;
    return r;
}

__device__ uint rgbaFloatToInt(float4 rgba)
{
    rgba.x = __saturatef(rgba.x);   // clamp to [0.0, 1.0]
    rgba.y = __saturatef(rgba.y);
    rgba.z = __saturatef(rgba.z);
    rgba.w = __saturatef(rgba.w);
    return (uint(rgba.w*255)<<24) | (uint(rgba.z*255)<<16) | (uint(rgba.y*255)<<8) | uint(rgba.x*255);
}

__device__ void addVisibility(float value, float3 pos, float depth)
{
	int w = sizeOfVolume.width, h = sizeOfVolume.height, d = sizeOfVolume.depth;
	//w = h = d = 32;
	//pos.x*0.5f + 0.5f, pos.y*0.5f + 0.5f, pos.z*0.5f + 0.5f
	int x = (int)((pos.x*0.5f + 0.5f) * w + 0.5f);
	x = (x >= w) ? (w - 1) : x;
	int y = (int)((pos.y*0.5f + 0.5f) * h + 0.5f);
	y = (y >= h) ? (h - 1) : y;
	int z = (int)((pos.z*0.5f + 0.5f) * d + 0.5f);
	z = (z >= d) ? (d - 1) : z;

	int index = z*w*h + y*w + x;
	
	atomicAdd((countVolume + index), 1);
	atomicAdd((visVolume + index), value);
	if (fabsf(depthVolume[index]) < 1e-6)
	{
		atomicAdd((depthVolume + index), depth);
	}
	
	//printf("atomicAdd %d \t %g \n", countVolume[index], visVolume[index]);

	float sample = tex3D(volTex, pos.x*0.5f + 0.5f, pos.y*0.5f + 0.5f, pos.z*0.5f + 0.5f);
	VolumeType intensity = (int)(sample + 0.5f);
	//float sample = tex3D(tex, pos.x*0.5f + 0.5f, pos.y*0.5f + 0.5f, pos.z*0.5f + 0.5f);
	//VolumeType intensity = (int)(sample*255 + 0.5f);
	atomicAdd((histogram + intensity), value);
}

__device__ void addVisibility2(float value, float3 pos)
{
	int w = sizeOfVolume.width, h = sizeOfVolume.height, d = sizeOfVolume.depth;
	//w = h = d = 32;
	//pos.x*0.5f + 0.5f, pos.y*0.5f + 0.5f, pos.z*0.5f + 0.5f
	int x = (int)((pos.x*0.5f + 0.5f) * w + 0.5f);
	x = (x >= w) ? (w - 1) : x;
	int y = (int)((pos.y*0.5f + 0.5f) * h + 0.5f);
	y = (y >= h) ? (h - 1) : y;
	int z = (int)((pos.z*0.5f + 0.5f) * d + 0.5f);
	z = (z >= d) ? (d - 1) : z;

	//int index = z*w*h + y*w + x;
	//atomicAdd((countVolume + index), 1);
	//atomicAdd((visVolume + index), value);

	float sample = tex3D(volTex, pos.x*0.5f + 0.5f, pos.y*0.5f + 0.5f, pos.z*0.5f + 0.5f);
	VolumeType intensity = (int)(sample + 0.5f);
	//float sample = tex3D(tex, pos.x*0.5f + 0.5f, pos.y*0.5f + 0.5f, pos.z*0.5f + 0.5f);
	//VolumeType intensity = (int)(sample*255 + 0.5f);
	atomicAdd((histogram2 + intensity), value);
}

__global__ void
surf_write(float *data, cudaExtent volumeSize)
{
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;
	int z = blockIdx.z*blockDim.z + threadIdx.z;

	if (x >= volumeSize.width || y >= volumeSize.height || z >= volumeSize.depth)
	{
		return;
	}
	float output = data[z*(volumeSize.width*volumeSize.height) + y*(volumeSize.width) + x];
	// surface writes need byte offsets for x!
	surf3Dwrite(output, volumeTexOut, x * sizeof(float), y, z);
}

__global__ void
tex_read(float x, float y, float z) {
	printf("x: %f, y: %f, z:%f, val: %f\n", x, y, z, tex3D(volumeTexIn, x, y, z));
}

__global__ void
d_render(uint *d_output, uint imageW, uint imageH,
         float density, float brightness,
         float transferOffset, float transferScale)
{
    const int maxSteps = 500;
    const float tstep = 0.01f;
    const float opacityThreshold = 0.95f;
    const float3 boxMin = make_float3(-1.0f, -1.0f, -1.0f);
    const float3 boxMax = make_float3(1.0f, 1.0f, 1.0f);

    uint x = blockIdx.x*blockDim.x + threadIdx.x;
    uint y = blockIdx.y*blockDim.y + threadIdx.y;

    if ((x >= imageW) || (y >= imageH)) return;

    float u = (x / (float) imageW)*2.0f-1.0f;
    float v = (y / (float) imageH)*2.0f-1.0f;

    // calculate eye ray in world space
    Ray eyeRay;
    eyeRay.o = make_float3(mul(c_invViewMatrix, make_float4(0.0f, 0.0f, 0.0f, 1.0f)));
    eyeRay.d = normalize(make_float3(u, v, -2.0f));
    eyeRay.d = mul(c_invViewMatrix, eyeRay.d);

    // find intersection with box
    float tnear, tfar;
    int hit = intersectBox(eyeRay, boxMin, boxMax, &tnear, &tfar);

    if (!hit) return;

    if (tnear < 0.0f) tnear = 0.0f;     // clamp to near plane

    // march along ray from front to back, accumulating color
    float4 sum = make_float4(0.0f);
    float t = tnear;
    float3 pos = eyeRay.o + eyeRay.d*tnear;
    float3 step = eyeRay.d*tstep;

    for (int i=0; i<maxSteps; i++)
    {
        // read from 3D texture
        // remap position to [0, 1] coordinates
        float sample = tex3D(tex, pos.x*0.5f+0.5f, pos.y*0.5f+0.5f, pos.z*0.5f+0.5f);
        //sample *= 64.0f;    // scale for 10-bit data

        // lookup in transfer function texture
        float4 col = tex1D(transferTex, (sample-transferOffset)*transferScale);
        col.w *= density;

        // "under" operator for back-to-front blending
        //sum = lerp(sum, col, col.w);

        // pre-multiply alpha
        col.x *= col.w;
        col.y *= col.w;
        col.z *= col.w;
        // "over" operator for front-to-back blending
        sum = sum + col*(1.0f - sum.w);

        // exit early if opaque
        if (sum.w > opacityThreshold)
            break;

        t += tstep;

        if (t > tfar) break;

        pos += step;
    }

    sum *= brightness;

    // write output color
    d_output[y*imageW + x] = rgbaFloatToInt(sum);
}

__global__ void
d_visibility(uint *d_output, uint imageW, uint imageH,
	float density, float brightness,
	float transferOffset, float transferScale)
{
	const int maxSteps = 500;
	const float tstep = 0.01f;
	const float opacityThreshold = 0.95f;
	const float3 boxMin = make_float3(-1.0f, -1.0f, -1.0f);
	const float3 boxMax = make_float3(1.0f, 1.0f, 1.0f);

	uint x = blockIdx.x*blockDim.x + threadIdx.x;
	uint y = blockIdx.y*blockDim.y + threadIdx.y;

	if ((x >= imageW) || (y >= imageH)) return;

	float u = (x / (float)imageW)*2.0f - 1.0f;
	float v = (y / (float)imageH)*2.0f - 1.0f;

	// calculate eye ray in world space
	Ray eyeRay;
	eyeRay.o = make_float3(mul(c_invViewMatrix, make_float4(0.0f, 0.0f, 0.0f, 1.0f)));
	eyeRay.d = normalize(make_float3(u, v, -2.0f));
	eyeRay.d = mul(c_invViewMatrix, eyeRay.d);

	// find intersection with box
	float tnear, tfar;
	int hit = intersectBox(eyeRay, boxMin, boxMax, &tnear, &tfar);

	if (!hit) return;

	if (tnear < 0.0f) tnear = 0.0f;     // clamp to near plane

										// march along ray from front to back, accumulating color
	float4 sum = make_float4(0.0f);
	float t = tnear;
	float3 pos = eyeRay.o + eyeRay.d*tnear;
	float3 step = eyeRay.d*tstep;

	for (int i = 0; i<maxSteps; i++)
	{
		// read from 3D texture
		// remap position to [0, 1] coordinates
		float sample = tex3D(tex, pos.x*0.5f + 0.5f, pos.y*0.5f + 0.5f, pos.z*0.5f + 0.5f);
		//sample *= 64.0f;    // scale for 10-bit data

		// lookup in transfer function texture
		float4 col = tex1D(transferTex, (sample - transferOffset)*transferScale);
		col.w *= density;

		// "under" operator for back-to-front blending
		//sum = lerp(sum, col, col.w);

		// pre-multiply alpha
		col.x *= col.w;
		col.y *= col.w;
		col.z *= col.w;
		// "over" operator for front-to-back blending

		float sumw = sum.w;
		sum = sum + col*(1.0f - sum.w);

		addVisibility(sum.w - sumw, pos, t - tnear);

		// exit early if opaque
		if (sum.w > opacityThreshold)
			break;

		t += tstep;

		if (t > tfar) break;

		pos += step;
	}

	sum *= brightness;

	// write output color
	d_output[y*imageW + x] = rgbaFloatToInt(sum);
}

__global__ void
d_visibilityLocal(uint *d_output, uint imageW, uint imageH,
	float density, float brightness,
	float transferOffset, float transferScale, int2 loc)
{
	const int maxSteps = 500;
	const float tstep = 0.01f;
	const float opacityThreshold = 0.95f;
	const float3 boxMin = make_float3(-1.0f, -1.0f, -1.0f);
	const float3 boxMax = make_float3(1.0f, 1.0f, 1.0f);

	uint x = blockIdx.x*blockDim.x + threadIdx.x;
	uint y = blockIdx.y*blockDim.y + threadIdx.y;

	if ((x >= imageW) || (y >= imageH)) return;

	float u = (x / (float)imageW)*2.0f - 1.0f;
	float v = (y / (float)imageH)*2.0f - 1.0f;

	// calculate eye ray in world space
	Ray eyeRay;
	eyeRay.o = make_float3(mul(c_invViewMatrix, make_float4(0.0f, 0.0f, 0.0f, 1.0f)));
	eyeRay.d = normalize(make_float3(u, v, -2.0f));
	eyeRay.d = mul(c_invViewMatrix, eyeRay.d);

	// find intersection with box
	float tnear, tfar;
	int hit = intersectBox(eyeRay, boxMin, boxMax, &tnear, &tfar);

	if (!hit) return;

	if (tnear < 0.0f) tnear = 0.0f;     // clamp to near plane

										// march along ray from front to back, accumulating color
	float4 sum = make_float4(0.0f);
	float t = tnear;
	float3 pos = eyeRay.o + eyeRay.d*tnear;
	float3 step = eyeRay.d*tstep;

	for (int i = 0; i<maxSteps; i++)
	{
		// read from 3D texture
		// remap position to [0, 1] coordinates
		float sample = tex3D(tex, pos.x*0.5f + 0.5f, pos.y*0.5f + 0.5f, pos.z*0.5f + 0.5f);
		//sample *= 64.0f;    // scale for 10-bit data

		// lookup in transfer function texture
		float4 col = tex1D(transferTex, (sample - transferOffset)*transferScale);
		col.w *= density;

		// "under" operator for back-to-front blending
		//sum = lerp(sum, col, col.w);

		// pre-multiply alpha
		col.x *= col.w;
		col.y *= col.w;
		col.z *= col.w;
		// "over" operator for front-to-back blending

		float sumw = sum.w;
		sum = sum + col*(1.0f - sum.w);

		addVisibility(sum.w - sumw, pos, t- tnear);
		
		// calculate visibility for selected region
		if (fabsf(x - loc.x) <= radius && fabsf(y - loc.y) <= radius)
		{
			addVisibility2(sum.w - sumw, pos);
		}

		// exit early if opaque
		if (sum.w > opacityThreshold)
			break;

		t += tstep;

		if (t > tfar) break;

		pos += step;
	}
	sum *= brightness;

	// write output color
	d_output[y*imageW + x] = rgbaFloatToInt(sum);
}

__global__ void
d_renderVisibility(uint *d_output, uint imageW, uint imageH,
	float density, float brightness,
	float transferOffset, float transferScale, int2 loc)
{
	const int maxSteps = 500;
	const float tstep = 0.01f;
	const float opacityThreshold = 0.95f;
	const float3 boxMin = make_float3(-1.0f, -1.0f, -1.0f);
	const float3 boxMax = make_float3(1.0f, 1.0f, 1.0f);

	uint x = blockIdx.x*blockDim.x + threadIdx.x;
	uint y = blockIdx.y*blockDim.y + threadIdx.y;

	if ((x >= imageW) || (y >= imageH)) return;

	float u = (x / (float)imageW)*2.0f - 1.0f;
	float v = (y / (float)imageH)*2.0f - 1.0f;

	// calculate eye ray in world space
	Ray eyeRay;
	eyeRay.o = make_float3(mul(c_invViewMatrix, make_float4(0.0f, 0.0f, 0.0f, 1.0f)));
	eyeRay.d = normalize(make_float3(u, v, -2.0f));
	eyeRay.d = mul(c_invViewMatrix, eyeRay.d);

	// find intersection with box
	float tnear, tfar;
	int hit = intersectBox(eyeRay, boxMin, boxMax, &tnear, &tfar);

	if (!hit)
	{
		d_output[y*imageW + x] = rgbaFloatToInt(make_float4(1.0f, 1.0f, 1.0f, 0.0f));
		return;
	}

	if (tnear < 0.0f) tnear = 0.0f;     // clamp to near plane

	// march along ray from front to back, accumulating color
	float4 sum = make_float4(0.0f);
	float t = tnear;
	float3 pos = eyeRay.o + eyeRay.d*tnear;
	float3 step = eyeRay.d*tstep;

	for (int i = 0; i<maxSteps; i++)
	{
		// read from 3D texture
		// remap position to [0, 1] coordinates
		float sample = tex3D(tex, pos.x*0.5f + 0.5f, pos.y*0.5f + 0.5f, pos.z*0.5f + 0.5f);
		//float vis = tex3D(visTex, pos.x*0.5f + 0.5f, pos.y*0.5f + 0.5f, pos.z*0.5f + 0.5f);
		//sample *= 64.0f;    // scale for 10-bit data

		// lookup in transfer function texture
		float4 col = tex1D(transferTex, (sample - transferOffset)*transferScale);
		//float4 col = make_float4(sample, sample, sample, sample);
		col.w *= density;
		//col.w /= vis;

		// "under" operator for back-to-front blending
		//sum = lerp(sum, col, col.w);

		// pre-multiply alpha
		col.x *= col.w;
		col.y *= col.w;
		col.z *= col.w;
		// "over" operator for front-to-back blending
		sum = sum + col*(1.0f - sum.w);

		// exit early if opaque
		if (sum.w > opacityThreshold)
			break;

		t += tstep;

		if (t > tfar) break;

		pos += step;
	}

	if (sum.w < 1.0f)
	{
		sum += make_float4(1.0f, 1.0f, 1.0f, 0.0f) * (1.0f - sum.w);
	}
	sum *= brightness;

	// draw selected region in inverted colors
	if (fabsf(x - loc.x) <= radius && fabsf(y - loc.y) <= radius)
	{
		auto w = sum.w;
		sum = make_float4(1, 1, 1, 1) - sum;
		sum.w = w;
	}

	// write output color
	d_output[y*imageW + x] = rgbaFloatToInt(sum);
}

extern "C"
void setTextureFilterMode(bool bLinearFilter)
{
    tex.filterMode = bLinearFilter ? cudaFilterModeLinear : cudaFilterModePoint;
}

inline void load_gaussians()
{
	FILE *gf1 = fopen("gaussian_5_5_5.txt", "r");
	int n = R5*R5*R5;
	for (int i = 0; i < n; i++)
	{
		fscanf(gf1, "%g", &g5[i]);
	}
	fclose(gf1);
	FILE *gf2 = fopen("gaussian_9_9_9.txt", "r");
	n = R9*R9*R9;
	for (int i = 0; i < n; i++)
	{
		fscanf(gf2, "%g", &g9[i]);
	}
	fclose(gf2);
}

__global__ void d_compute_saliency()
{
	uint x = blockIdx.x*blockDim.x + threadIdx.x;
	uint y = blockIdx.y*blockDim.y + threadIdx.y;
	uint z = blockIdx.z*blockDim.z + threadIdx.z;
	printf("%d %d %d \n", x, y, z);
}

extern "C"
void gaussian(float *lch_volume, cudaExtent volumeSize, float *out)
{
	auto len = volumeSize.width * volumeSize.height * volumeSize.depth;
	int w = volumeSize.width, h = volumeSize.height, d = volumeSize.depth;
	const int r1 = R5 / 2;
	const int r2 = R9 / 2;
	int w2 = w - r2;
	int h2 = h - r2;
	int d2 = d - r2;
	for (int z = r2; z < d2; z++)
	{
		for (int y = r2; y < h2; y++)
		{
			for (int x = r2; x < w2; x++)
			{
				int index = z*w*h + y*w + x;
				float sum9 = 0, sum5 = 0;
				for (int i = -r2; i <= r2; i++)
				{
					for (int j = -r2; j <= r2; j++)
					{
						for (int k = -r2; k <= r2; k++)
						{
							int idx = (z + i)*w*h + (y + j)*w + (x + k);
							sum9 += g9[(i + r2)*R9*R9 + (j + r2)*R9 + (k + r2)] * ((float *)lch_volume)[idx];
						}
					}
				}
				for (int i = -r1; i <= r1; i++)
				{
					for (int j = -r1; j <= r1; j++)
					{
						for (int k = -r1; k <= r1; k++)
						{
							int idx = (z + i)*w*h + (y + j)*w + (x + k);
							sum5 += g5[(i + r1)*R5*R5 + (j + r1)*R5 + (k + r1)] * ((float*)lch_volume)[idx];
						}
					}
				}
				out[index] = abs(sum5 - sum9);
			}
		}
	}
}

extern "C"
void compute_saliency()
{
	//std::cout << "compute_saliency()" << std::endl;
	dim3 blockSize3(16, 16, 16);
	dim3 gridSize3 = dim3(iDivUp(sizeOfVolume.width, blockSize3.x), iDivUp(sizeOfVolume.height, blockSize3.y), iDivUp(sizeOfVolume.depth, blockSize3.z));

	auto len = sizeOfVolume.width * sizeOfVolume.height * sizeOfVolume.depth;
	int w = sizeOfVolume.width, h = sizeOfVolume.height, d = sizeOfVolume.depth;
	memset(saliencyVolume, 0, sizeof(float) * len);

	float4 lch_array[BIN_COUNT] = { 0 };
	for (int i = 0; i < BIN_COUNT; i++)
	{
		float4 lch=rgb_to_lch(tf_array[i]);
		lch_array[i].x = lch.x;
		lch_array[i].y = lch.y;
		lch_array[i].z = lch.z;
		lch_array[i].w = lch.w;
	}

	float *lightness = (float *)malloc(len * sizeof(float));
	float *chroma = (float *)malloc(len * sizeof(float));
	float *g1 = (float *)malloc(len * sizeof(float));
	float *g2 = (float *)malloc(len * sizeof(float));
	memset(g1, 0, len * sizeof(float));
	memset(g2, 0, len * sizeof(float));

	//for (int z = 0; z < d; z++)
	//{
	//	for (int y = 0; y < h; y++)
	//	{
	//		for (int x = 0; x < w; x++)
	//		{
	//			int index = z*w*h + y*w + x;
	//			int intensity = (int)raw_volume[index];
	//			lightness[index] = lch_array[intensity].x;
	//			chroma[index] = lch_array[intensity].y;
	//		}
	//	}
	//}
	for (int i = 0; i < len; i++)
	{
		int intensity = (int)raw_volume[i];
		lightness[i] = lch_array[intensity].x;
		chroma[i] = lch_array[intensity].y;
	}

	gaussian(lightness, sizeOfVolume, g1);
	gaussian(chroma, sizeOfVolume, g2);

	//for (int z = 0; z < d; z++)
	//{
	//	for (int y = 0; y < h; y++)
	//	{
	//		for (int x = 0; x < w; x++)
	//		{
	//			int index = z*w*h + y*w + x;
	//			saliencyVolume[index] = (g1[index] + g2[index]) / 2;
	//			vwsVolume[index] = saliencyVolume[index] * visVolume[index];
	//		}
	//	}
	//}
	for (int i = 0; i < len; i++)
	{
		saliencyVolume[i] = (g1[i] + g2[i]) / 2;
		//vwsVolume[i] = saliencyVolume[i] * visVolume[i];
	}

	free(g1);
	free(g2);
	free(lightness);
	free(chroma);
}

extern "C"
void compute_vws()
{
	auto len = sizeOfVolume.width * sizeOfVolume.height * sizeOfVolume.depth;
	int w = sizeOfVolume.width, h = sizeOfVolume.height, d = sizeOfVolume.depth;
	memset(vwsVolume, 0, sizeof(float) * len);
	for (int i = 0; i < len; i++)
	{
		vwsVolume[i] = saliencyVolume[i] * visVolume[i];
	}
}

extern "C"
void compute_saliency_once()
{
	if (!saliency_once)
	{
		saliency_once = true;
		compute_saliency();
	}
}

extern "C"
void initCuda(void *h_volume, cudaExtent volumeSize)
{
	auto len = volumeSize.width * volumeSize.height * volumeSize.depth;
	checkCudaErrors(cudaMallocManaged(&raw_volume, sizeof(VolumeType) * len));
	memcpy(raw_volume, h_volume, sizeof(VolumeType) * len);
	//auto cube = malloc(sizeof(float) * len);
	//memset(cube, 0, sizeof(float) * len);
	//printf("%g\n", *((float*)cube+len-1));

	load_gaussians();
	checkCudaErrors(cudaMallocManaged(&saliencyVolume, sizeof(float) * len));
	checkCudaErrors(cudaMallocManaged(&vwsVolume, sizeof(float) * len));
	checkCudaErrors(cudaMallocManaged(&featureVolume, sizeof(unsigned char) * len));

	compute_saliency();
	compute_vws();

	sizeOfVolume = volumeSize;
	printf("volumeSize \t %d %d %d\n", sizeOfVolume.width, sizeOfVolume.height, sizeOfVolume.depth);
	//printf("volumeSize \t %d %d %d\n", volumeSize.width, volumeSize.height, volumeSize.depth);

	//auto len = volumeSize.width * volumeSize.height * volumeSize.depth;
	checkCudaErrors(cudaMallocManaged(&depthVolume, sizeof(float) * len));
	checkCudaErrors(cudaMallocManaged(&visVolume, sizeof(float) * len));
	checkCudaErrors(cudaMallocManaged(&countVolume, sizeof(int) * len));
	//printf("%g\n", *(visVolume + 1));
	//printf("%d\n", *(countVolume + 1));

	//auto tf2 = tf_array;
	//printf("sizeof \t histogram %d \t tf_array %d \t tf2 %d %d \n", sizeof(histogram) / sizeof(float), sizeof(tf_array) / sizeof(float4), sizeof(tf2), sizeof(float4));

	cudaChannelFormatDesc channelDesc0 = cudaCreateChannelDesc<VisibilityType>();
	//checkCudaErrors(cudaMalloc3DArray(&d_visibilityArray, &channelDesc0, sizeOfVolume, cudaArraySurfaceLoadStore));
	checkCudaErrors(cudaMalloc3DArray(&d_visibilityArray, &channelDesc0, sizeOfVolume));
	//std::cout << "channelDesc0 \t" << channelDesc0.x << "\t" << channelDesc0.y << "\t" << channelDesc0.z << "\t" << channelDesc0.w << "\t" << channelDesc0.f << std::endl;

    // create 3D array
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<VolumeType>();
    checkCudaErrors(cudaMalloc3DArray(&d_volumeArray, &channelDesc, volumeSize));

    // copy data to 3D array
    cudaMemcpy3DParms copyParams = {0};
    copyParams.srcPtr   = make_cudaPitchedPtr(h_volume, volumeSize.width*sizeof(VolumeType), volumeSize.width, volumeSize.height);
    copyParams.dstArray = d_volumeArray;
    copyParams.extent   = volumeSize;
    copyParams.kind     = cudaMemcpyHostToDevice;
    checkCudaErrors(cudaMemcpy3D(&copyParams));

    // set texture parameters
    tex.normalized = true;                      // access with normalized texture coordinates
    tex.filterMode = cudaFilterModeLinear;      // linear interpolation
    tex.addressMode[0] = cudaAddressModeClamp;  // clamp texture coordinates
    tex.addressMode[1] = cudaAddressModeClamp;

    // bind array to 3D texture
    checkCudaErrors(cudaBindTextureToArray(tex, d_volumeArray, channelDesc));

	// set texture parameters
	volTex.normalized = true;                      // access with normalized texture coordinates
	//volTex.filterMode = cudaFilterModeLinear;      // linear interpolation
	volTex.filterMode = cudaFilterModePoint;      // nearest-neighbor interpolation
	volTex.addressMode[0] = cudaAddressModeClamp;  // clamp texture coordinates
	volTex.addressMode[1] = cudaAddressModeClamp;

	// bind array to 3D texture
	checkCudaErrors(cudaBindTextureToArray(volTex, d_volumeArray, channelDesc));

    // create transfer function texture
    float4 transferFunc[] =
    {
        {  0.0, 0.0, 0.0, 0.0, },
        {  1.0, 0.0, 0.0, 1.0, },
        {  1.0, 0.5, 0.0, 1.0, },
        {  1.0, 1.0, 0.0, 1.0, },
        {  0.0, 1.0, 0.0, 1.0, },
        {  0.0, 1.0, 1.0, 1.0, },
        {  0.0, 0.0, 1.0, 1.0, },
        {  1.0, 0.0, 1.0, 1.0, },
        {  0.0, 0.0, 0.0, 0.0, },
    };

    cudaChannelFormatDesc channelDesc2 = cudaCreateChannelDesc<float4>();
    cudaArray *d_transferFuncArray;
    //checkCudaErrors(cudaMallocArray(&d_transferFuncArray, &channelDesc2, sizeof(transferFunc)/sizeof(float4), 1));
    //checkCudaErrors(cudaMemcpyToArray(d_transferFuncArray, 0, 0, transferFunc, sizeof(transferFunc), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMallocArray(&d_transferFuncArray, &channelDesc2, sizeof(tf_array) / sizeof(float4), 1));
	checkCudaErrors(cudaMemcpyToArray(d_transferFuncArray, 0, 0, tf_array, sizeof(tf_array), cudaMemcpyHostToDevice));

    transferTex.filterMode = cudaFilterModeLinear;
    transferTex.normalized = true;    // access with normalized texture coordinates
    transferTex.addressMode[0] = cudaAddressModeClamp;   // wrap texture coordinates

    // Bind the array to the texture
	checkCudaErrors(cudaBindTextureToArray(transferTex, d_transferFuncArray, channelDesc2));
}

extern "C"
void freeCudaBuffers()
{
    checkCudaErrors(cudaFreeArray(d_volumeArray));
	checkCudaErrors(cudaFreeArray(d_transferFuncArray));
	checkCudaErrors(cudaFreeArray(d_visibilityArray));
	checkCudaErrors(cudaFree(depthVolume));
	checkCudaErrors(cudaFree(visVolume));
	checkCudaErrors(cudaFree(countVolume));
}

extern "C"
void render_visibility_kernel(dim3 gridSize, dim3 blockSize, uint *d_output, uint imageW, uint imageH,
	float density, float brightness, float transferOffset, float transferScale)
{
	auto len = sizeOfVolume.width * sizeOfVolume.height * sizeOfVolume.depth;
	//auto cube = malloc(sizeof(float) * len);
	//memset(visVolume, 0, sizeof(VisibilityType) * len);
	cudaMemset(countVolume, 0, sizeof(int) * len);
	cudaMemset(depthVolume, 0, sizeof(float) * len);
	cudaMemset(visVolume, 0, sizeof(VisibilityType) * len);
	cudaMemset(histogram, 0, sizeof(float)*BIN_COUNT);
	cudaMemset(histogram2, 0, sizeof(float)*BIN_COUNT);

	//d_visibility << <gridSize, blockSize >> >(d_output, imageW, imageH, density, brightness, transferOffset, transferScale);
	//cudaDeviceSynchronize();

	d_visibility<< <gridSize, blockSize >> >(d_output, imageW, imageH, density, brightness, transferOffset, transferScale);
	cudaDeviceSynchronize();
}

extern "C"
void render_kernel(dim3 gridSize, dim3 blockSize, uint *d_output, uint imageW, uint imageH,
                   float density, float brightness, float transferOffset, float transferScale, int2 loc)
{
    //d_render<<<gridSize, blockSize>>>(d_output, imageW, imageH, density, brightness, transferOffset, transferScale);

	auto len = sizeOfVolume.width * sizeOfVolume.height * sizeOfVolume.depth;
	//auto cube = malloc(sizeof(float) * len);
	//memset(visVolume, 0, sizeof(VisibilityType) * len);
	cudaMemset(countVolume, 0, sizeof(int) * len);
	cudaMemset(depthVolume, 0, sizeof(float) * len);
	cudaMemset(visVolume, 0, sizeof(VisibilityType) * len);
	cudaMemset(histogram, 0, sizeof(float)*BIN_COUNT);
	cudaMemset(histogram2, 0, sizeof(float)*BIN_COUNT);

	//d_visibility << <gridSize, blockSize >> >(d_output, imageW, imageH, density, brightness, transferOffset, transferScale);
	//cudaDeviceSynchronize();

	d_visibilityLocal << <gridSize, blockSize >> >(d_output, imageW, imageH, density, brightness, transferOffset, transferScale, loc);
	cudaDeviceSynchronize();

	// copy data to 3D array
	cudaMemcpy3DParms copyParams2 = { 0 };
	copyParams2.srcPtr = make_cudaPitchedPtr(visVolume, sizeOfVolume.width * sizeof(VisibilityType), sizeOfVolume.width, sizeOfVolume.height);
	copyParams2.dstArray = d_visibilityArray;
	copyParams2.extent = sizeOfVolume;
	copyParams2.kind = cudaMemcpyHostToDevice;
	checkCudaErrors(cudaMemcpy3D(&copyParams2));

	//checkCudaErrors(cudaMemcpyToArrayAsync(d_visibilityArray, 0, 0, countVolume, volumeSize.width*volumeSize.height*sizeof(int), cudaMemcpyHostToDevice));

	// set texture parameters
	visTex.normalized = true;                      // access with normalized texture coordinates
	//visTex.filterMode = cudaFilterModeLinear;      // linear interpolation
	visTex.filterMode = cudaFilterModePoint;      // linear interpolation
	visTex.addressMode[0] = cudaAddressModeClamp;  // clamp texture coordinates
	visTex.addressMode[1] = cudaAddressModeClamp;

	cudaChannelFormatDesc channelDesc0 = cudaCreateChannelDesc<VisibilityType>();
	checkCudaErrors(cudaBindTextureToArray(visTex, d_visibilityArray, channelDesc0));

	d_renderVisibility << <gridSize, blockSize >> >(d_output, imageW, imageH, density, brightness, transferOffset, transferScale, loc);

	cudaDeviceSynchronize();

	if (get_apply())
	{
		set_apply(false);
		printf("loc %d %d\n", loc.x, loc.y);
		blend_tf_rgba(make_float3(g_SelectedColor[0], g_SelectedColor[1], g_SelectedColor[2]));
	}

	if (get_gaussian())
	{
		set_gaussian(false);
		printf("loc %d %d\n", loc.x, loc.y);
		gaussian_tf(make_float3(g_SelectedColor[0], g_SelectedColor[1], g_SelectedColor[2]));
	}

	if (get_save())
	{
		set_save(false);

		char buffer[_MAX_PATH];
		sprintf(buffer, "~%s", volume_file);
		printf("save a visibility field and histograms to %s.\n", buffer);

		auto fp = fopen(buffer, "wb");
		fwrite(visVolume, sizeof(VisibilityType), len, fp);
		fclose(fp);

		{
			sprintf(buffer, "~%s.saliency.raw", volume_file);
			auto fp = fopen(buffer, "wb");
			fwrite(saliencyVolume, sizeof(float), len, fp);
			fclose(fp);
		}

		{
			sprintf(buffer, "~%s.vws.raw", volume_file);
			auto fp = fopen(buffer, "wb");
			fwrite(vwsVolume, sizeof(float), len, fp);
			fclose(fp);
		}

		{
			sprintf(buffer, "~%s.feature.raw", volume_file);
			auto fp = fopen(buffer, "wb");
			fwrite(featureVolume, sizeof(unsigned char), len, fp);
			fclose(fp);
		}

		sprintf(buffer, "~%s.depth.raw", volume_file);
		auto fp0 = fopen(buffer, "wb");
		fwrite(depthVolume, sizeof(float), len, fp0);
		fclose(fp0);

		sprintf(buffer, "~%s.data.txt", volume_file);
		auto fp6 = fopen(buffer, "w");
		for (int z = 0;z < sizeOfVolume.depth;z+=2)
		{
			for (int y = 0;y < sizeOfVolume.height;y+=2)
			{
				for (int x = 0;x < sizeOfVolume.width;x+=2)
				{
					int i = z*sizeOfVolume.width * sizeOfVolume.height + y*sizeOfVolume.width + x;
					fprintf(fp6, "{%f,%f,%f}\n", raw_volume[i] / 255.f, depthVolume[i], visVolume[i]);
				}
			}
		}
		fclose(fp6);

		sprintf(buffer, "~%s.txt", volume_file);
		auto fp1 = fopen(buffer, "w");
		for (int i = 0; i < BIN_COUNT; i++)
		{
			fprintf(fp1, "%f\n", histogram[i]);
		}
		fclose(fp1);

		printf("loc %d %d\n", loc.x, loc.y);
		sprintf(buffer, "~%s.2.txt", volume_file);
		auto fp2 = fopen(buffer, "w");
		for (int i = 0; i < BIN_COUNT; i++)
		{
			fprintf(fp2, "%f\n", histogram2[i]);
		}
		fclose(fp2);

		sprintf(buffer, "~%s.3.txt", volume_file);
		auto fp3 = fopen(buffer, "w");
		for (int i = 0; i < BIN_COUNT; i++)
		{
			fprintf(fp3, "%f\n", histogram3[i]);
		}
		fclose(fp3);

		sprintf(buffer, "~%s.4.txt", volume_file);
		auto fp4 = fopen(buffer, "w");
		for (int i = 0; i < BIN_COUNT; i++)
		{
			fprintf(fp4, "%f\n", histogram4[i]);
		}
		fclose(fp4);

		sprintf(buffer, "~%s.tf.txt", volume_file);
		auto fp5 = fopen(buffer, "w");
		for (int i = 0; i < BIN_COUNT; i++)
		{
			fprintf(fp5, "{%f,%f,%f,%f}\n", tf_array[i].x, tf_array[i].y, tf_array[i].z, tf_array[i].w);
		}
		fclose(fp5);
	}

	if (get_discard())
	{
		set_discard(false);
		restore_tf();
	}

	if (get_backup())
	{
		set_backup(false);
		backup_tf();
	}
}

extern "C"
void copyInvViewMatrix(float *invViewMatrix, size_t sizeofMatrix)
{
    checkCudaErrors(cudaMemcpyToSymbol(c_invViewMatrix, invViewMatrix, sizeofMatrix));
}


#endif // #ifndef _VOLUMERENDER_KERNEL_CU_
