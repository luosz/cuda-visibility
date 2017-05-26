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

#include <helper_cuda.h>
#include <helper_cuda.h>
#ifndef _VOLUMERENDER_KERNEL_CU_
#define _VOLUMERENDER_KERNEL_CU_

#include <helper_cuda.h>
#include <helper_math.h>

#include <iostream>
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

__device__ float *visVolume = NULL;
__device__ int *countVolume = NULL;
//__device__ __managed__ cudaExtent sizeOfVolume;// = make_cudaExtent(32, 32, 32);

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

__device__ void addVisibility(float value, float3 pos)
{
	int w, h, d;
	w = h = d = 32;
	int x = (int)((pos.x + 1) * 0.5f * w + 0.5f);
	x = (x >= w) ? (w - 1) : x;
	int y = (int)((pos.y + 1) * 0.5f * h + 0.5f);
	y = (y >= h) ? (h - 1) : y;
	int z = (int)((pos.z + 1) * 0.5f * d + 0.5f);
	z = (z >= d) ? (d - 1) : z;

	int index = z*w*h + y*w + x;
	
	//countVolume[index] += 1;
	//visVolume[index] += value;
	atomicAdd((countVolume + index), 1);
	atomicAdd((visVolume + index), value);
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
		sum = sum + col*(1.0f - sum.w);

		addVisibility(sum.w, pos);

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

extern "C"
void setTextureFilterMode(bool bLinearFilter)
{
    tex.filterMode = bLinearFilter ? cudaFilterModeLinear : cudaFilterModePoint;
}

extern "C"
void initCuda(void *h_volume, cudaExtent volumeSize)
{
	auto len = volumeSize.width * volumeSize.height * volumeSize.depth;
	auto cube = malloc(sizeof(float) * len);
	memset(cube, 0, sizeof(float) * len);
	printf("%g\n", *((float*)cube+len-1));

	//sizeOfVolume = volumeSize;
	//printf("%d %d %d\n", sizeOfVolume.width, sizeOfVolume.height, sizeOfVolume.depth);
	//printf("%d %d %d\n", volumeSize.width, volumeSize.height, volumeSize.depth);

	//auto len = volumeSize.width * volumeSize.height * volumeSize.depth;
	checkCudaErrors(cudaMallocManaged(&visVolume, sizeof(float) * len));
	checkCudaErrors(cudaMallocManaged(&countVolume, sizeof(int) * len));
	printf("%g\n", *(visVolume + 1));
	printf("%d\n", *(countVolume + 1));

    // create 3D array
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<VolumeType>();
    checkCudaErrors(cudaMalloc3DArray(&d_volumeArray, &channelDesc, volumeSize));

	std::cout << "channelDesc\t" << channelDesc.x << "\t" << channelDesc.y << "\t" << channelDesc.z << "\t" << channelDesc.w << "\t" << channelDesc.f << std::endl;
	std::cout << "volumeSize\t" << volumeSize.width << "\t" << volumeSize.height << "\t" << volumeSize.depth << std::endl;

	auto channelDesc0 = cudaCreateChannelDesc<float>();
	checkCudaErrors(cudaMalloc3DArray(&d_visibilityArray, &channelDesc0, volumeSize, cudaArraySurfaceLoadStore));
	std::cout << "channelDesc\t" << channelDesc0.x << "\t" << channelDesc0.y << "\t" << channelDesc0.z << "\t" << channelDesc0.w << "\t" << channelDesc0.f << std::endl;

	auto ptr = make_cudaPitchedPtr(d_visibilityArray, volumeSize.width * sizeof(VolumeType), volumeSize.width, volumeSize.height);
	//checkCudaErrors(cudaMemset3D(ptr, 0, volumeSize));

    // copy data to 3D array
    cudaMemcpy3DParms copyParams = {0};
    copyParams.srcPtr   = make_cudaPitchedPtr(h_volume, volumeSize.width*sizeof(VolumeType), volumeSize.width, volumeSize.height);
    copyParams.dstArray = d_volumeArray;
    copyParams.extent   = volumeSize;
    copyParams.kind     = cudaMemcpyHostToDevice;
    checkCudaErrors(cudaMemcpy3D(&copyParams));

	auto copyParams2 = copyParams;
	copyParams2.srcPtr = make_cudaPitchedPtr(cube, volumeSize.width * sizeof(float), volumeSize.width, volumeSize.height);
	copyParams2.dstArray = d_visibilityArray;
	checkCudaErrors(cudaMemcpy3D(&copyParams2));

	volumeTexIn.filterMode = cudaFilterModeLinear;
	checkCudaErrors(cudaBindSurfaceToArray(volumeTexOut, d_visibilityArray));

	dim3 blockSize(8, 8, 8);
	dim3 gridSize((volumeSize.width + 7) / 8, (volumeSize.height + 7) / 8, (volumeSize.depth + 7) / 8);
	//surf_write << <gridSize, blockSize >> >((float *)cube, volumeSize);

	checkCudaErrors(cudaBindTextureToArray(volumeTexIn, d_visibilityArray));

	//tex_read << <1, 1 >> >(1.5, 1.5, 1.5);

	//checkCudaErrors(cudaDeviceSynchronize());

    // set texture parameters
    tex.normalized = true;                      // access with normalized texture coordinates
    tex.filterMode = cudaFilterModeLinear;      // linear interpolation
    tex.addressMode[0] = cudaAddressModeClamp;  // clamp texture coordinates
    tex.addressMode[1] = cudaAddressModeClamp;

    // bind array to 3D texture
    checkCudaErrors(cudaBindTextureToArray(tex, d_volumeArray, channelDesc));

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
    checkCudaErrors(cudaMallocArray(&d_transferFuncArray, &channelDesc2, sizeof(transferFunc)/sizeof(float4), 1));
    checkCudaErrors(cudaMemcpyToArray(d_transferFuncArray, 0, 0, transferFunc, sizeof(transferFunc), cudaMemcpyHostToDevice));

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
	checkCudaErrors(cudaFree(visVolume));
	checkCudaErrors(cudaFree(countVolume));
}


extern "C"
void render_kernel(dim3 gridSize, dim3 blockSize, uint *d_output, uint imageW, uint imageH,
                   float density, float brightness, float transferOffset, float transferScale)
{
    //d_render<<<gridSize, blockSize>>>(d_output, imageW, imageH, density, brightness, transferOffset, transferScale);

	d_visibility << <gridSize, blockSize >> >(d_output, imageW, imageH, density, brightness, transferOffset, transferScale);
	cudaDeviceSynchronize();
}

extern "C"
void copyInvViewMatrix(float *invViewMatrix, size_t sizeofMatrix)
{
    checkCudaErrors(cudaMemcpyToSymbol(c_invViewMatrix, invViewMatrix, sizeofMatrix));
}


#endif // #ifndef _VOLUMERENDER_KERNEL_CU_
