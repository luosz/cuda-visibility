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

/*
    Volume rendering sample

    This sample loads a 3D volume from disk and displays it using
    ray marching and 3D textures.

    Note - this is intended to be an example of using 3D textures
    in CUDA, not an optimized volume renderer.

    Changes
    sgg 22/3/2010
    - updated to use texture for display instead of glDrawPixels.
    - changed to render from front-to-back rather than back-to-front.
*/

#pragma once

// OpenGL Graphics includes
#include <helper_gl.h>
#if defined (__APPLE__) || defined(MACOSX)
  #pragma clang diagnostic ignored "-Wdeprecated-declarations"
  #include <GLUT/glut.h>
  #ifndef glutCloseFunc
  #define glutCloseFunc glutWMCloseFunc
  #endif
#else
#include <GL/freeglut.h>
#endif

// CUDA Runtime, Interop, and includes
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <cuda_profiler_api.h>
#include <vector_types.h>
#include <vector_functions.h>
#include <driver_functions.h>

// CUDA utilities
#include <helper_cuda.h>

// Helper functions
#include <helper_cuda.h>
#include <helper_functions.h>
#include <helper_timer.h>
#include <helper_math.h>

#include <arrayfire.h>
#include <af/util.h>

#include <iostream>
#include <sstream>
#include <fstream>
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <ctime>
#include <cctype>
#include <cstring>
//#include <AntTweakBar.h>
#include <FreeImage.h>
#include "tinyxml2.h"
#include "util.h"
#include "def.h"
#include "kmeans.h"

// include cereal for serialization
#include "cereal/archives/xml.hpp"
#include "serialize.h"

#include "ColorSpace/ColorSpace.h"

using namespace std;

#include "mainwindow.h"
#include <QtWidgets/QApplication>

#if defined(_WIN32) || defined(_WIN64)
//  MiniGLUT.h is provided to avoid the need of having GLUT installed to 
//  recompile this example. Do not use it in your own programs, better
//  install and use the actual GLUT library SDK.
#   define USE_MINI_GLUT
#endif

#if defined(USE_MINI_GLUT)
#   include "../src/MiniGLUT.h"
#elif defined(_MACOSX)
#   include <GLUT/glut.h>
#else
#   include <GL/glut.h>
#endif

typedef unsigned int uint;
typedef unsigned char uchar;

#define MAX_EPSILON_ERROR 5.00f
#define THRESHOLD         0.30f

// Define the files that are to be save and the reference images for validation
const char *sOriginal[] =
{
    "volume.ppm",
    NULL
};

const char *sReference[] =
{
    "ref_volume.ppm",
    NULL
};

const char *sSDKsample = "CUDA 3D Volume Render";

//const char *volumeFilename = "Bucky.raw";
//cudaExtent volumeSize = make_cudaExtent(32, 32, 32);
//typedef unsigned char VolumeType;

//char *volumeFilename = "mrt16_angio.raw";
//cudaExtent volumeSize = make_cudaExtent(416, 512, 112);
//typedef unsigned short VolumeType;

//const char *tfs[] = { "nucleon_naive_proportional_2.tfi","vortex_naive_proportional_2.tfi","CT-Knee_spectrum_6.tfi","E_1324_Rainbow6_even_2.tfi" };
const char *tfs[] = { "nucleon_naive_proportional.tfi","vortex_naive_proportional.tfi","CT-Knee_spectrum_6.tfi","Rainbow3_even.tfi" };
const char *volumes[] = { "nucleon.raw","vorts1.raw","CT-Knee.raw","E_1324.raw" };
const int data_index = 0;
const char *tfFile = tfs[data_index];
const char *volumeFilename = volumes[data_index];
char volumeFilename_buffer[_MAX_PATH];

/**
41, 41, 41
128, 128, 128
379, 229, 305
432, 432, 432
*/
const size_t sizes[] = {
41, 41, 41,
128, 128, 128,
379, 229, 305,
432, 432, 432 };
cudaExtent volumeSize = make_cudaExtent(sizes[0+data_index*3], sizes[1+data_index*3], sizes[2+data_index*3]);
typedef unsigned char VolumeType;

char filename_buffer[_MAX_PATH] = "";

std::vector<float> intensity_list;
std::vector<float4> rgba_list;
std::vector<float4> rgba_list_backup;
std::vector<int> peak_indices;
std::string program_path;
std::vector<string> volume_list;
std::vector<string> volume_list2;

float gaussian5[R5*R5*R5] = { 0 };
float gaussian9[R9*R9*R9] = { 0 };

int2 loc = {0, 0};
int2 loc2 = {0, 0};
float2 locf = {0,0};
float2 diff = {0,0};
bool dragMode = false; // mouse tracking mode
bool time_varying_tf_editing = D_TIME_VARYING_TF_EDITING;
bool time_varying_tf_reset = D_TIME_VARYING_TF_RESET;
bool time_varying_vws_optimization = D_TIME_VARYING_VWS_OPTIMIZATION;
bool temporal_visibility = D_TERMPORAL_VISIBILITY;

//uint width = 512, height = 512;
uint width = D_WIDTH, height = D_HEIGHT;
dim3 blockSize(16, 16);
dim3 gridSize;

extern "C" unsigned int get_width()
{
	return width;
}

extern "C" unsigned int get_height()
{
	return height;
}


//dim3 blockSize3(16, 16, 16);
//dim3 gridSize3;

float3 viewRotation;
float3 viewTranslation = make_float3(0.0, 0.0, -4.0f);
float invViewMatrix[12];
const char *view_file = "~view.xml";

float density = 0.05f;
float brightness = 1.0f;
float transferOffset = 0.0f;
float transferScale = 1.0f;
bool linearFiltering = true;

GLuint pbo = 0;     // OpenGL pixel buffer object
GLuint tex = 0;     // OpenGL texture object
struct cudaGraphicsResource *cuda_pbo_resource; // CUDA Graphics Resource (to transfer PBO)

StopWatchInterface *timer = 0;

// Auto-Verification Code
const int frameCheckNumber = 2;
int fpsCount = 0;        // FPS count for averaging
int fpsLimit = 1;        // FPS limit for sampling
int g_Index = 0;
unsigned int frameCount = 0;

int *pArgc;
char **pArgv;

#ifndef MAX
#define MAX(a,b) ((a > b) ? a : b)
#endif

int screenshot_id = -1;

extern "C" int get_screenshot_id()
{
	return screenshot_id;
}

extern "C" int get_next_screenshot_id(int id)
{
	return (id + 1) % 4;
}

extern "C" int increase_screenshot_id()
{
	return screenshot_id = get_next_screenshot_id(screenshot_id);
}

MainWindow *qt_window = NULL;

extern bool get_display_selection();
extern void set_display_selection(bool value);

extern "C" void set_temporal_tf(bool value);
extern "C" void set_kmeans(bool value);
//extern "C" bool get_apply();
extern "C" void set_apply(bool value);
//extern "C" bool get_save();
extern "C" void set_save(bool value);
//extern "C" bool get_discard();
extern "C" void set_discard(bool value);
//extern "C" bool get_gaussian();
extern "C" void set_gaussian(bool value);
//extern "C" bool get_backup();
extern "C" void set_backup(bool value);

extern "C" void reset_temporal_visibility_histogram();
extern "C" void update_volume(void *h_volume, cudaExtent volumeSize);
extern "C" void bind_tf_texture();
extern "C" VolumeType * get_raw_volume();
extern "C" unsigned char* get_feature_volume();
extern "C" float* get_vws_volume();
extern "C" int get_feature_number();
extern "C" void set_feature_number(int val);
extern "C" float* get_feature_array();
extern "C" float* get_feature_vws_array();
extern "C" void gaussian(float *lch_volume, cudaExtent volumeSize, float *out);
extern "C" void compute_saliency();
extern "C" void compute_vws();
extern "C" void compute_saliency_once();
//extern "C" bool get_accumulate_visibility();
extern "C" void set_accumulate_visibility(bool value);

typedef float(*Pointer)[4];
extern "C" Pointer get_SelectedColor();
extern "C" void set_SelectedColor(float r, float g, float b);
extern "C" bool* get_ApplyColor();
extern "C" bool* get_ApplyAlpha();
extern "C" int get_region_size();
extern "C" float4* get_tf_array();

extern "C" int get_bin_count();
extern "C" void set_volume_file(const char *file, int n);
extern "C" void backup_tf();
extern "C" void restore_tf();
extern "C" void render_visibility_kernel(dim3 gridSize, dim3 blockSize, uint *d_output, uint imageW, uint imageH,
	float density, float brightness, float transferOffset, float transferScale);

extern "C" void setTextureFilterMode(bool bLinearFilter);
extern "C" void initCuda(void *h_volume, cudaExtent volumeSize);
extern "C" void freeCudaBuffers();
extern "C" void render_kernel(dim3 gridSize, dim3 blockSize, uint *d_output, uint imageW, uint imageH,
                              float density, float brightness, float transferOffset, float transferScale, int2 loc);
extern "C" void copyInvViewMatrix(float *invViewMatrix, size_t sizeofMatrix);

extern "C" void set_save_rendering(bool value);
extern "C" void set_save_ppm(bool value);

extern "C" void erase_ppm_on_gpu();
extern "C" void load_ppm_to_gpu(const char *file = "../kmeans/~k32_~screenshot_0.ppm");

void apply_save_rendering()
{
	set_save_rendering(true);
}

void apply_tf_editing()
{
	//std::cout << "apply_tf_editing()" << std::endl;
	set_gaussian(true);
}

void apply_temporal_tf_editing()
{
	set_temporal_tf(true);
}

void reset_transfer_function()
{
	//std::cout << "reset_transfer_function()" << std::endl;
	set_discard(true);
}

/// RGB to LCH color conversion
extern "C" float4 rgb_to_lch(float4 rgba)
{
	ColorSpace::Rgb rgb(rgba.x, rgba.y, rgba.z);
	ColorSpace::Lch lch;
	ColorSpace::LchConverter::ToColorSpace(&rgb, &lch);
	return make_float4(lch.l, lch.c, lch.h, rgba.w);
}

std::string log_filename()
{
	return "~log_time-varying.txt";
}

void initPixelBuffer();

/// Test cases for searching for features
void search_feature_test(float intensity = 0)
{
	cout << "features" << endl;
	float *feature_array = get_feature_array();
	int count = get_feature_number();
	int n = count << 1;
	for (int i = 0; i < n; i += 2)
	{
		cout << feature_array[i] << ends << feature_array[i + 1] << endl;
	}
	cout << "--------" << endl;
	cout << "binary search tests " << endl;
	int i0 = binary_search(feature_array, 0, n, intensity);
	cout << intensity << ends << i0 << endl;
	for (int i=0;i<n;i++)
	{
		float x = feature_array[i] + 0.01f;
		int i1 = binary_search(feature_array, 0, n, x);
		cout << x << ends << i1 << ends << feature_array[i1] << endl;
	}
	cout << "--------" << endl;
}

/// Find what feature does intensity belong to.
/// Return -1 if the intensity is not in a feature interval.
int search_feature(float intensity = 0)
{
	float *features = get_feature_array();
	int count = get_feature_number();
	int n = count << 1;
	//int idx = linear_search(features, 0, n, intensity);
	int idx = binary_search(features, 0, n, intensity);
	if (idx != -1 && (idx & 1) == 0)
	{
		return idx / 2 + 1;
	}
	return 0;
}

extern "C"
void compute_feature_volume()
{
	auto p = get_raw_volume();
	if (!p)
	{
		std::cerr << "get_raw_volume() returns NULL!" << std::endl;
		return;
	}

	auto featureVolume = get_feature_volume();
	auto len = volumeSize.width * volumeSize.height * volumeSize.depth;
	int w = volumeSize.width;
	int h = volumeSize.height;
	int d = volumeSize.depth;
	memset(featureVolume, 0, sizeof(unsigned char) * len);
	int n = 0;
	//std::cout << w << std::ends << h << std::ends << d << std::ends << len << std::endl;
	//ofstream info("c:/work/log_feature.txt");
	//ofstream info2("c:/work/log_intensity.txt");
	for (int i = 0; i < len; i++)
	{
		//int index = z*w*h + y*w + x;
		//auto intensity = p[i];
		float intensity = p[i] / (float)255;
		auto idx = (unsigned char)search_feature(intensity);
		//info2 << i << " " << (int)p[i] << " " << (int)idx << std::endl;
		featureVolume[i] = idx;
		//if (idx > 0)
		//{
		//	n++;
		//	//info << i << " " << intensity << " " << (int)idx << std::endl;
		//}
	}
	//info.close();
	//info2.close();

	//std::cout << std::endl << n << std::endl;
}

/// Compute the array of feature Visibility-Weighted Saliency scores.
void compute_vws_array()
{
	auto p = get_raw_volume();
	if (!p)
	{
		std::cerr << "get_raw_volume() returns NULL!" << std::endl;
		return;
	}
	float *feature_vws_array = get_feature_vws_array();
	float *vws_volume = get_vws_volume();
	auto featureVolume = get_feature_volume();
	int count = get_feature_number();
	auto len = volumeSize.width * volumeSize.height * volumeSize.depth;
	int w = volumeSize.width;
	int h = volumeSize.height;
	int d = volumeSize.depth;

	memset(feature_vws_array, 0, D_BIN_COUNT * sizeof(float));

	//std::cout << "compute_vws_array \n";
	int n = 0;

	//ofstream info("c:/work/log_vws.txt");

	for (int i = 0;i < len;i++)
	{
		auto idx = featureVolume[i];
		if (idx > 0)
		{
			n++;
			feature_vws_array[idx - 1] += vws_volume[i];
			//info << i << " " << (int)idx << " " << vws_volume[i] << std::endl;
		}
	}

	//info.close();

	//std::cout << std::endl << n << std::endl;

	//for (int i=0;i<count;i++)
	//{
	//	std::cout << feature_vws_array[i] << std::ends;
	//}
	//std::cout << std::endl;
	float sum = 0;
	for (int i = 0;i < count;i++)
	{
		//feature_vws[i] = feature_vws_array[i];
		sum += feature_vws_array[i];
	}
	if (abs(sum) > 0)
	{
		for (int i = 0;i < count;i++)
		{
			feature_vws_array[i] /= sum;
		}
	}
}

void render();
void render_visibility();
void load_lookuptable(std::vector<float> intensity, std::vector<float4> rgba);

/// VWS transfer function optimization
std::vector<float> vws_tf_optimization()
{
	const float stepsize = 0.05f;
	const float epsilon = 0.0001f;
	const int margin = 4;
	float mrms = 0;
	int mindex = 0;

	int count = get_feature_number();
	std::vector <float> target(count);

	// targets: equal weights e.g. 1/3, 1/3, 1/3
	for (int i=0;i<count;i++)
	{
		target[i] = 1.f / count;
	}

	//target[0] = 0.1f;
	//target[1] = 0.3f;
	//target[2] = 0.6f;

	float *feature_vws_array = get_feature_vws_array();
	auto start = std::clock();
	compute_saliency();
	compute_feature_volume();
	compute_vws();
	compute_vws_array();
	auto end = std::clock();
	std::cout << "compute saliency, visibility and vws duration (seconds): " << (end - start) / (double)CLOCKS_PER_SEC << std::endl;
	int iteration = 0;
	const int MAX_LOOP = 40;

	std::cout << "count=" << count << std::endl;
	float rms = 0;
	for (int i = 0; i < count; i++)
	{
		rms += (feature_vws_array[i] - target[i])*(feature_vws_array[i] - target[i]);
		std::cout << feature_vws_array[i] << "\t" << target[i] << std::endl;
	}
	mrms = rms;
	std::cout << "rms=" << rms << std::endl;

	//// string to be saved to log file
	//std::stringstream ss;
	//ss << iteration << "\t" << rms << "\t" << count;
	//for (int i = 0; i < count; i++)
	//{
	//	ss << "\t" << feature_vws_array[i];
	//}
	//ss << std::endl;

	start = std::clock();

	while (rms > epsilon && iteration < MAX_LOOP && mindex + margin >= iteration)
	{
		++iteration;

		// peaks, steps
		// update alpha of peak control points with gradient*step
		for (int i = 0; i < count; i++)
		{
			float gradient = 2 * (feature_vws_array[i] - target[i]);
			float step = -gradient * stepsize;
			float peak = rgba_list[peak_indices[i]].w + step;
			peak = peak < 0 ? 0 : (peak > 1 ? 1 : peak);
			rgba_list[peak_indices[i]].w = peak;
		}

		load_lookuptable(intensity_list, rgba_list);
		bind_tf_texture();
		render_visibility();
		compute_vws();
		compute_vws_array();

		// objective function
		rms = 0;
		for (int i = 0; i < count; i++)
		{
			rms += (feature_vws_array[i] - target[i])*(feature_vws_array[i] - target[i]);
		}
		if (abs(rms) > 0)
		{
			rms /= count;
		}
		else
		{
			std::cerr << "Error: rms=0" << std::endl;
		}
		if (rms < mrms)
		{
			mrms = rms;
			mindex = iteration;
		}

		//ss << iteration << "\t" << rms << "\t" << count;
		//for (int i=0;i<count;i++)
		//{
		//	ss << "\t" << feature_vws_array[i];
		//}
		//ss << std::endl;
	}

	end = std::clock();
	float duration = (end - start) / (float)CLOCKS_PER_SEC;
	std::cout << "gradient descent optimization duration (seconds): " << duration << std::endl;

	//ofstream out("~log.txt");
	//out << ss.str();
	//out.close();

	std::cout << "target \n";
	for (int i = 0; i < count; i++)
	{
		std::cout << target[i] << "\t";
	}
	std::cout << std::endl;
	std::cout << "feature_vws_array \n";
	for (int i = 0; i < count; i++)
	{
		std::cout << feature_vws_array[i] << "\t";
	}
	std::cout << std::endl;

	std::cout << "iteration " << iteration << "\t rms=" << rms << std::endl;

	std::vector<float> ans = { duration, (float)iteration, rms };
	return ans;
}

/// VWS transfer function optimization
std::vector<float> vws_tf_optimization_linesearch()
{
	const float stepsize = 0.05f;
	const float epsilon = 0.0001f;
	const int margin = 4;
	float mrms = 0;
	int mindex = 0;

	int count = get_feature_number();
	std::vector <float> target(count);

	// targets: equal weights e.g. 1/3, 1/3, 1/3
	for (int i = 0; i < count; i++)
	{
		target[i] = 1.f / count;
	}

	//target[0] = 0.1f;
	//target[1] = 0.3f;
	//target[2] = 0.6f;

	//float sum = 0;
	//for (int i=0;i<count;i++)
	//{
	//	//target[i] = rgba_list[peak_indices[i]].w;
	//	target[i] = intensity_list[peak_indices[i]];
	//	sum += target[i];
	//}	
	//if (sum>0)
	//{
	//	for (int i = 0; i < count; i++)
	//	{
	//		target[i] /= sum;
	//	}
	//}
	//else
	//{
	//	std::cerr << "Error: sum of targets is zero" << std::endl;
	//}

	float *feature_vws_array = get_feature_vws_array();
	auto start = std::clock();
	compute_saliency();
	compute_feature_volume();
	compute_vws();
	compute_vws_array();
	auto end = std::clock();
	std::cout << "compute saliency, visibility and vws duration (seconds): " << (end - start) / (double)CLOCKS_PER_SEC << std::endl;
	int iteration = 0;
	const int MAX_LOOP = 40;

	std::cout << "count=" << count << std::endl;
	float rms = 0;
	for (int i = 0; i < count; i++)
	{
		rms += (feature_vws_array[i] - target[i])*(feature_vws_array[i] - target[i]);
		std::cout << feature_vws_array[i] << "\t" << target[i] << std::endl;
	}
	mrms = rms;
	std::cout << "rms=" << rms << std::endl;

	std::stringstream ss;
	ss << iteration << "\t" << rms << "\t" << count;
	for (int i = 0; i < count; i++)
	{
		ss << "\t" << feature_vws_array[i];
	}
	ss << std::endl;

	start = std::clock();

	while (rms > epsilon && iteration < MAX_LOOP && mindex + margin >= iteration)
	{
		++iteration;
		std::vector<float> gradients;
		std::vector<float> peak_list;

		// peaks, steps
		// update alpha of peak control points with gradient*step
		for (int i = 0; i < count; i++)
		{
			float gradient = 2 * (feature_vws_array[i] - target[i]);
			gradients.push_back(gradient);
			float step = -gradient * stepsize;
			float peak = rgba_list[peak_indices[i]].w + step;
			peak = peak < 0 ? 0 : (peak > 1 ? 1 : peak);
			rgba_list[peak_indices[i]].w = peak;
			peak_list.push_back(peak);
		}

		load_lookuptable(intensity_list, rgba_list);
		bind_tf_texture();
		render_visibility();
		compute_vws();
		compute_vws_array();

		// objective function
		rms = 0;
		for (int i = 0; i < count; i++)
		{
			rms += (feature_vws_array[i] - target[i])*(feature_vws_array[i] - target[i]);
		}
		if (abs(rms) > 0)
		{
			rms /= count;
		}
		else
		{
			std::cerr << "Error: rms=0" << std::endl;
		}

		// line search
		std::vector<int> multipliers = {2, 4, 8};
		for (int i : multipliers)
		{
			for (int i = 0; i < count; i++)
			{
				float step = -gradients[i] * stepsize * i;
				float peak = rgba_list[peak_indices[i]].w + step;
				peak = peak < 0 ? 0 : (peak > 1 ? 1 : peak);
				rgba_list[peak_indices[i]].w = peak;
			}

			load_lookuptable(intensity_list, rgba_list);
			bind_tf_texture();
			render_visibility();
			compute_vws();
			compute_vws_array();

			// objective function
			float rms2 = 0;
			for (int i = 0; i < count; i++)
			{
				rms2 += (feature_vws_array[i] - target[i])*(feature_vws_array[i] - target[i]);
			}
			if (abs(rms2) > 0)
			{
				rms2 /= count;
			}
			else
			{
				std::cerr << "Error: rms=0" << std::endl;
			}

			// update minimum rms and peak list
			if (rms2 < rms)
			{
				rms = rms2;
				for (int i = 0; i < count; i++)
				{
					peak_list[i] = rgba_list[peak_indices[i]].w;
				}
			}
		}

		// after line search, set peaks to the best step along the gradient
		for (int i = 0; i < count; i++)
		{
			rgba_list[peak_indices[i]].w = peak_list[i];
		}

		if (rms < mrms)
		{
			mrms = rms;
			mindex = iteration;
		}

		ss << iteration << "\t" << rms << "\t" << count;
		for (int i = 0; i < count; i++)
		{
			ss << "\t" << feature_vws_array[i];
		}
		ss << std::endl;
	}

	end = std::clock();
	float duration = (end - start) / (float)CLOCKS_PER_SEC;
	std::cout << "gradient descent with line search optimization duration (seconds): " << duration << std::endl;

	ofstream out("~log.txt");
	out << ss.str();
	out.close();

	std::cout << "target \n";
	for (int i = 0; i < count; i++)
	{
		std::cout << target[i] << "\t";
	}
	std::cout << std::endl;
	std::cout << "feature_vws_array \n";
	for (int i = 0; i < count; i++)
	{
		std::cout << feature_vws_array[i] << "\t";
	}
	std::cout << std::endl;

	std::cout << "iteration " << iteration << "\t rms=" << rms << std::endl;

	std::vector<float> ans = { duration, (float)iteration, rms };
	return ans;
}

/// Count how many features are defined in the transfer function
void count_features(std::vector<float> intensity, std::vector<float4> rgba)
{
	peak_indices.clear();
	float epsilon = std::numeric_limits<float>::epsilon();
	float *features = get_feature_array();
	int count = 0;
	for (int i = 1; i < intensity.size() - 1; i++)
	{
		if (rgba[i].w > 0 && abs(rgba[i-1].w) < epsilon && abs(rgba[i+1].w) < epsilon)
		{
			features[count++] = intensity[i - 1];
			features[count++] = intensity[i + 1];
			peak_indices.push_back(i);
		}
	}
	set_feature_number(count>>1);
}

/// Load a transfer function into a lookup table for rendering
void load_lookuptable(std::vector<float> intensity, std::vector<float4> rgba)
{
	auto n = get_bin_count();
	float4 *tf = get_tf_array();
	int last = (int)intensity.size() - 1;
	int k = 0;
	for (int i = 0; i < n; i++)
	{
		auto d = i / (float)(n - 1);
		while (k <= last && d > intensity[k])
		{
			k++;
		}
		if (k == 0)
		{
			tf[i] = rgba[k];
		}
		else
		{
			if (k > last)
			{
				tf[i] = rgba[last];
			}
			else
			{
				auto a = intensity[k - 1];
				auto b = intensity[k];
				if (abs(b - a) > 0.0001f)
				{
					auto t = (d - a) / (b - a);
					tf[i] = lerp(rgba[k - 1], rgba[k], t);
				}
				else
				{
					tf[i] = rgba[k];
				}
			}
		}
	}
}

/// Open Voreen transfer functions
void openTransferFunctionFromVoreenXML(const char *filename)
{
	//ui->statusBar->showMessage(QString(filename));

	tinyxml2::XMLDocument doc;
	auto r = doc.LoadFile(filename);
	
	if (r != tinyxml2::XML_SUCCESS)
	{
		std::cout << "failed to open file" << endl;
		return;
	}

	auto transFuncIntensity = doc.FirstChildElement("VoreenData")->FirstChildElement("TransFuncIntensity");

	//auto threshold = transFuncIntensity->FirstChildElement("threshold");
	//if (threshold != NULL)
	//{
	//	Threshold_x(atof(threshold->Attribute("x")));
	//	Threshold_y(atof(threshold->Attribute("y")));
	//}
	//else
	//{
	//	Threshold_x(atof(transFuncIntensity->FirstChildElement("lower")->Attribute("value")));
	//	Threshold_y(atof(transFuncIntensity->FirstChildElement("upper")->Attribute("value")));
	//}

	//auto domain = transFuncIntensity->FirstChildElement("domain");
	//if (domain != NULL)
	//{
	//	Domain_x(atof(domain->Attribute("x")));
	//	Domain_y(atof(domain->Attribute("y")));
	//	std::cout << "domain x=" << Domain_x() << " y=" << Domain_y() << std::endl;
	//}
	//else
	//{
	//	Domain_x(0);
	//	Domain_y(1);
	//	std::cout << "domain doesn't exist. default: " << Domain_x() << " " << Domain_y() << std::endl;
	//}

	auto key = doc.FirstChildElement("VoreenData")->FirstChildElement("TransFuncIntensity")->FirstChildElement("Keys")->FirstChildElement("key");

	//float4 transferFunc[] = { 0 };
	//std::vector<float> intensity_list;
	//std::vector<float4> rgba_list;
	//intensity_list_clear();
	//colour_list_clear();
	//opacity_list_clear();

	//std::vector<float> intensity_list;
	//std::vector<float4> rgba_list;
	intensity_list.clear();
	rgba_list.clear();

	do
	{
		auto intensity = (float)atof(key->FirstChildElement("intensity")->Attribute("value"));
		//intensity_list_push_back(intensity);
		intensity_list.push_back(intensity);
		int r = atoi(key->FirstChildElement("colorL")->Attribute("r"));
		int g = atoi(key->FirstChildElement("colorL")->Attribute("g"));
		int b = atoi(key->FirstChildElement("colorL")->Attribute("b"));
		int a = atoi(key->FirstChildElement("colorL")->Attribute("a"));
		//std::vector<double> colour;
		//colour.push_back(normalise_rgba(r));
		//colour.push_back(normalise_rgba(g));
		//colour.push_back(normalise_rgba(b));
		//colour_list_push_back(colour);
		//opacity_list_push_back(normalise_rgba(a));
		rgba_list.push_back(make_float4(normalise_rgba(r), normalise_rgba(g), normalise_rgba(b), normalise_rgba(a)));

		bool split = (0 == strcmp("true", key->FirstChildElement("split")->Attribute("value")));
		//std::cout << "intensity=" << intensity;
		//std::cout << "\tsplit=" << (split ? "true" : "false");
		//std::cout << "\tcolorL r=" << r << " g=" << g << " b=" << b << " a=" << a;
		const auto epsilon = 1e-6f;
		if (split)
		{
			//intensity_list_push_back(intensity + epsilon);
			intensity_list.push_back(intensity+ epsilon);
			auto colorR = key->FirstChildElement("colorR");
			int r2 = atoi(colorR->Attribute("r"));
			int g2 = atoi(colorR->Attribute("g"));
			int b2 = atoi(colorR->Attribute("b"));
			int a2 = atoi(colorR->Attribute("a"));
			//std::vector<double> colour2;
			//colour2.push_back(normalise_rgba(r2));
			//colour2.push_back(normalise_rgba(g2));
			//colour2.push_back(normalise_rgba(b2));
			//colour2.push_back(normalise_rgba(a2));
			//colour_list_push_back(colour2);
			//opacity_list_push_back(normalise_rgba(a2));
			rgba_list.push_back(make_float4(normalise_rgba(r2), normalise_rgba(g2), normalise_rgba(b2), normalise_rgba(a2)));
			//std::cout << "\tcolorR r=" << r2 << " g=" << g2 << " b=" << b2 << " a=" << a2;
		}
		//std::cout << endl;

		key = key->NextSiblingElement();
	} while (key);

	//for (int i=0;i<intensity_list.size();i++)
	//{
	//	printf("%g\n", intensity_list[i]);
	//}

	count_features(intensity_list, rgba_list);
	//search_feature_test();
	load_lookuptable(intensity_list, rgba_list);
	backup_tf();
}

void saveTransferFunctionToVoreenXML(const char *filename)
{
	tinyxml2::XMLDocument doc;

	auto declaration = doc.NewDeclaration();
	doc.InsertEndChild(declaration);
	auto voreenData = doc.NewElement("VoreenData");
	voreenData->SetAttribute("version", 1);
	auto transFuncIntensity = doc.NewElement("TransFuncIntensity");
	transFuncIntensity->SetAttribute("type", "TransFuncIntensity");

	// add alphaMode
	auto alphaMode = doc.NewElement("alphaMode");
	alphaMode->SetAttribute("value", 1);
	transFuncIntensity->InsertEndChild(alphaMode);

	// add gammaValue
	auto gammaValue = doc.NewElement("gammaValue");
	gammaValue->SetAttribute("value", 1);
	transFuncIntensity->InsertEndChild(gammaValue);

	// add domain
	auto domain = doc.NewElement("domain");
	domain->SetAttribute("x", 0);
	domain->SetAttribute("y", 1);
	transFuncIntensity->InsertEndChild(domain);

	// add threshold
	auto threshold = doc.NewElement("threshold");
	threshold->SetAttribute("x", 0);
	threshold->SetAttribute("y", 1);
	transFuncIntensity->InsertEndChild(threshold);

	//intensity_list
	//rgba_list

	int n = intensity_list.size();

	// add Keys
	auto keys = doc.NewElement("Keys");
	for (int i = 0; i < n; i++)
	{
		auto key = doc.NewElement("key");
		key->SetAttribute("type", "TransFuncMappingKey");
		auto intensity = doc.NewElement("intensity");
		intensity->SetAttribute("value", intensity_list[i]);
		auto split = doc.NewElement("split");
		split->SetAttribute("value", "false");
		auto colorL = doc.NewElement("colorL");
		
		colorL->SetAttribute("r", denormalise_rgba(rgba_list[i].x));
		colorL->SetAttribute("g", denormalise_rgba(rgba_list[i].y));
		colorL->SetAttribute("b", denormalise_rgba(rgba_list[i].z));
		colorL->SetAttribute("a", denormalise_rgba(rgba_list[i].w));
		key->InsertEndChild(intensity);
		key->InsertEndChild(split);
		key->InsertEndChild(colorL);
		keys->InsertEndChild(key);
	}
	transFuncIntensity->InsertEndChild(keys);

	voreenData->InsertEndChild(transFuncIntensity);
	doc.InsertEndChild(voreenData);

	auto r = doc.SaveFile(filename);
	if (r != tinyxml2::XML_SUCCESS)
	{
		std::cout << "failed to save file " << filename << endl;
	}
}

void save_tf_array_to_voreen_XML(const char *filename)
{
	tinyxml2::XMLDocument doc;

	auto declaration = doc.NewDeclaration();
	doc.InsertEndChild(declaration);
	auto voreenData = doc.NewElement("VoreenData");
	voreenData->SetAttribute("version", 1);
	auto transFuncIntensity = doc.NewElement("TransFuncIntensity");
	transFuncIntensity->SetAttribute("type", "TransFuncIntensity");

	// add alphaMode
	auto alphaMode = doc.NewElement("alphaMode");
	alphaMode->SetAttribute("value", 1);
	transFuncIntensity->InsertEndChild(alphaMode);

	// add gammaValue
	auto gammaValue = doc.NewElement("gammaValue");
	gammaValue->SetAttribute("value", 1);
	transFuncIntensity->InsertEndChild(gammaValue);

	// add domain
	auto domain = doc.NewElement("domain");
	domain->SetAttribute("x", 0);
	domain->SetAttribute("y", 1);
	transFuncIntensity->InsertEndChild(domain);

	// add threshold
	auto threshold = doc.NewElement("threshold");
	threshold->SetAttribute("x", 0);
	threshold->SetAttribute("y", 1);
	transFuncIntensity->InsertEndChild(threshold);

	auto n = get_bin_count();
	float4 *tf = get_tf_array();

	// add Keys
	auto keys = doc.NewElement("Keys");
	for (int i = 0; i < n; i++)
	{
		auto key = doc.NewElement("key");
		key->SetAttribute("type", "TransFuncMappingKey");
		auto intensity = doc.NewElement("intensity");
		intensity->SetAttribute("value", i / 255.f);
		auto split = doc.NewElement("split");
		split->SetAttribute("value", "false");
		auto colorL = doc.NewElement("colorL");

		auto rgba = tf[i];
		colorL->SetAttribute("r", denormalise_rgba(rgba.x));
		colorL->SetAttribute("g", denormalise_rgba(rgba.y));
		colorL->SetAttribute("b", denormalise_rgba(rgba.z));
		colorL->SetAttribute("a", denormalise_rgba(rgba.w));
		key->InsertEndChild(intensity);
		key->InsertEndChild(split);
		key->InsertEndChild(colorL);
		keys->InsertEndChild(key);
	}
	transFuncIntensity->InsertEndChild(keys);

	voreenData->InsertEndChild(transFuncIntensity);
	doc.InsertEndChild(voreenData);

	auto r = doc.SaveFile(filename);
	if (r != tinyxml2::XML_SUCCESS)
	{
		std::cout << "failed to save file " << filename << endl;
	}
}

inline void set_temporal_visibility(bool v)
{
	temporal_visibility = v;
	if (temporal_visibility)
	{
		time_varying_vws_optimization = time_varying_tf_reset = time_varying_tf_editing = false;
	}
	if (qt_window)
	{
		qt_window->update_checkbox();
	}
}

inline void add_volume_to_list_for_update_vortex()
{
	volume_list.clear();
	volume_list2.clear();
	set_temporal_visibility(false);
	char file[_MAX_PATH];
	for (int i=99;i>=1;i--)
	{
		sprintf(file, "vorts%d.raw", i);
		volume_list.push_back(file);
	}
	rgba_list_backup = rgba_list;
	diff.x = ((float)loc2.x - loc.x) / volume_list.size();
	diff.y = ((float)loc2.y - loc.y) / volume_list.size();
}

inline void add_volume_to_list_for_update_reverse()
{
	volume_list.clear();
	for (auto i = volume_list2.rbegin(); i != volume_list2.rend(); i++)
	{
		volume_list.push_back(*i);
	}
	volume_list2.clear();
	set_temporal_visibility(true);
	rgba_list_backup = rgba_list;
	diff.x = ((float)loc2.x - loc.x) / volume_list.size();
	diff.y = ((float)loc2.y - loc.y) / volume_list.size();
}

inline void add_volume_to_list_for_update_supernova()
{
	volume_list.clear();
	volume_list2.clear();
	set_accumulate_visibility(false);
	char file[_MAX_PATH];
	for (int i = 1354; i >= 1295; i--)
	{
		sprintf(file, "E_%d.raw", i);
		volume_list.push_back(file);
	}
	rgba_list_backup = rgba_list;
	diff.x = ((float)loc2.x - loc.x) / volume_list.size();
	diff.y = ((float)loc2.y - loc.y) / volume_list.size();
}

inline void add_volume_to_list_for_update_test()
{
	std::vector<std::string> str = {"vorts1.raw","vorts1 - Copy.raw","vorts1 - Copy (2).raw","vorts1 - Copy (3).raw","vorts1 - Copy (4).raw"};
	volume_list.clear();
	volume_list2.clear();
	set_temporal_visibility(false);
	for (auto i:str)
	{
		volume_list.push_back(i);
	}
	rgba_list_backup = rgba_list;
	diff.x = ((float)loc2.x - loc.x) / volume_list.size();
	diff.y = ((float)loc2.y - loc.y) / volume_list.size();
}

extern "C" void add_volume_to_list_for_update_from_vector(std::vector<std::string> filelist)
{
	volume_list.clear();
	volume_list2.clear();
	set_accumulate_visibility(false);
	for (auto rit = filelist.rbegin(); rit != filelist.rend(); ++rit)
	{
		volume_list.push_back(*rit);
	}
	rgba_list_backup = rgba_list;
	diff.x = ((float)loc2.x - loc.x) / volume_list.size();
	diff.y = ((float)loc2.y - loc.y) / volume_list.size();
}

std::vector<float> optimize_for_a_frame()
{
	int count = get_feature_number();
	std::vector<float> ans = vws_tf_optimization();
	std::cout << "Peak control points\n";
	for (auto i:peak_indices)
	{
		std::cout << rgba_list[i].w << "\t" << rgba_list_backup[i].w << std::endl;
	}
	rgba_list = rgba_list_backup;
	return ans;
}

void *loadRawFile(const char *filename, size_t size);

extern "C" void load_mhd_file(std::string filename)
{
	strcpy(volumeFilename_buffer, filename.c_str());
	volumeFilename = volumeFilename_buffer;
	set_volume_file(volumeFilename, strlen(volumeFilename));

	// load volume data
	char *path = sdkFindFilePath(volumeFilename, program_path.c_str());
	//printf("volume %s\n", path);

	if (path == 0)
	{
		printf("Error finding file '%s'\n", volumeFilename);
		exit(EXIT_FAILURE);
	}

	std::cout<<path<<std::endl;

	int w=-1, h=-1, d=-1;
	std::string raw_file;
	std::string path2 = path;
	std::ifstream infile(path);
	if (infile.is_open())
	{
		std::string line;
		while (std::getline(infile, line))
		{
			std::size_t found = line.find("DimSize");
			if (found != std::string::npos)
			{
				std::size_t equal = line.find("=");
				if (equal != std::string::npos)
				{
					stringstream ss(line.substr(equal+1));
					ss>>w>>h>>d;
					volumeSize = make_cudaExtent(w, h, d);
				}
			}

			found = line.find("ElementDataFile");
			if (found != std::string::npos)
			{
				std::size_t equal = line.find("=");
				if (equal != std::string::npos)
				{
					stringstream ss(line.substr(equal + 1));
					ss>>raw_file;
					auto pos = path2.find(volumeFilename);
					if (pos != std::string::npos)
					{
						path2.replace(pos, strlen(volumeFilename), raw_file);
						std::cout<<path2<<std::endl;
					}
				}
			}
		}
	}
	infile.close();

	size_t size = volumeSize.width*volumeSize.height*volumeSize.depth * sizeof(VolumeType);
	void *h_volume = loadRawFile(path2.c_str(), size);
	initCuda(h_volume, volumeSize);
	free(h_volume);
}

void load_a_volume_and_optimize()
{
	if (!volume_list.empty())
	{
		strcpy(volumeFilename_buffer, volume_list.back().c_str());
		volumeFilename = volumeFilename_buffer;
		set_volume_file(volumeFilename, strlen(volumeFilename));

		// load volume data
		char *path = sdkFindFilePath(volumeFilename, program_path.c_str());
		//printf("volume %s\n", path);

		if (path == 0)
		{
			printf("Error finding file '%s'\n", volumeFilename);
			exit(EXIT_FAILURE);
		}

		size_t size = volumeSize.width*volumeSize.height*volumeSize.depth * sizeof(VolumeType);
		void *h_volume = loadRawFile(path, size);

		//initCuda(h_volume, volumeSize);
		update_volume(h_volume, volumeSize);

		//// optimize for a frame
		//auto ans = optimize_for_a_frame();
		//std::stringstream ss;
		//std::cout << volumeFilename;
		//ss << volumeFilename;
		//for (auto i:ans)
		//{
		//	std::cout << "\t" << i;
		//	ss << "\t" << i;
		//}
		//std::cout << std::endl;
		//ss << std::endl;
		//ofstream out(log_filename(), std::ios_base::app);
		//out << ss.str();
		//out.close();

		// vws optimiztion for a frame
		if (time_varying_vws_optimization)
		{
			auto ans = optimize_for_a_frame();
			std::stringstream ss;
			std::cout << volumeFilename;
			ss << volumeFilename;
			for (auto i : ans)
			{
				std::cout << "\t" << i;
				ss << "\t" << i;
			}
			std::cout << std::endl;
			ss << std::endl;
			ofstream out(log_filename(), std::ios_base::app);
			out << ss.str();
			out.close();
		}
		else
		{
			// apply tf editing
			if (time_varying_tf_reset)
			{
				reset_transfer_function();
			}
			if (time_varying_tf_editing)
			{
				locf.x += diff.x;
				locf.y += diff.y;
				loc.x = (int)locf.x;
				loc.y = (int)locf.y;
				apply_tf_editing();
			}
			if(temporal_visibility)
			{
				set_accumulate_visibility(true);
			}
		}

		free(h_volume);
		volume_list2.push_back(volume_list.back());
		volume_list.pop_back();    

		if (volume_list.empty() && temporal_visibility)
		{
			reset_temporal_visibility_histogram();
		}

		if (qt_window)
		{
			qt_window->delay_draw_transfer_function_and_visibility_histograms();
		}
	}
}

void computeFPS()
{
    frameCount++;
    fpsCount++;

    if (fpsCount == fpsLimit)
    {
        char fps[256];
        float ifps = 1.f / (sdkGetAverageTimerValue(&timer) / 1000.f);
		sprintf(fps, "Volume Render: %3.1f fps", ifps);
        //sprintf(fps, "CUDA volume rendering: %3.1f fps", ifps);

        glutSetWindowTitle(fps);
        fpsCount = 0;

        fpsLimit = (int)MAX(1.f, ifps);
        sdkResetTimer(&timer);
    }
}

void render_visibility()
{
	copyInvViewMatrix(invViewMatrix, sizeof(float4) * 3);

	// map PBO to get CUDA device pointer
	uint *d_output;
	// map PBO to get CUDA device pointer
	checkCudaErrors(cudaGraphicsMapResources(1, &cuda_pbo_resource, 0));
	size_t num_bytes;
	checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&d_output, &num_bytes,
		cuda_pbo_resource));
	//printf("CUDA mapped PBO: May access %ld bytes\n", num_bytes);

	// clear image
	checkCudaErrors(cudaMemset(d_output, 0, width*height * 4));

	// call CUDA kernel, writing results to PBO
	render_visibility_kernel(gridSize, blockSize, d_output, width, height, density, brightness, transferOffset, transferScale);

	getLastCudaError("kernel failed");

	checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0));
}

// render image using CUDA
void render()
{
	// update volume for time-varying data
	load_a_volume_and_optimize();

    copyInvViewMatrix(invViewMatrix, sizeof(float4)*3);

    // map PBO to get CUDA device pointer
    uint *d_output;
    // map PBO to get CUDA device pointer
    checkCudaErrors(cudaGraphicsMapResources(1, &cuda_pbo_resource, 0));
    size_t num_bytes;
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&d_output, &num_bytes,
                                                         cuda_pbo_resource));
    //printf("CUDA mapped PBO: May access %ld bytes\n", num_bytes);

    // clear image
    checkCudaErrors(cudaMemset(d_output, 0, width*height*4));

    // call CUDA kernel, writing results to PBO
    render_kernel(gridSize, blockSize, d_output, width, height, density, brightness, transferOffset, transferScale, loc);

    getLastCudaError("kernel failed");

    checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0));
}

// display results using OpenGL (called by GLUT)
void display()
{
    sdkStartTimer(&timer);

    // use OpenGL to build view matrix
    GLfloat modelView[16];
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();
    glRotatef(-viewRotation.x, 1.0, 0.0, 0.0);
    glRotatef(-viewRotation.y, 0.0, 1.0, 0.0);
    glTranslatef(-viewTranslation.x, -viewTranslation.y, -viewTranslation.z);
    glGetFloatv(GL_MODELVIEW_MATRIX, modelView);
    glPopMatrix();

    invViewMatrix[0] = modelView[0];
    invViewMatrix[1] = modelView[4];
    invViewMatrix[2] = modelView[8];
    invViewMatrix[3] = modelView[12];
    invViewMatrix[4] = modelView[1];
    invViewMatrix[5] = modelView[5];
    invViewMatrix[6] = modelView[9];
    invViewMatrix[7] = modelView[13];
    invViewMatrix[8] = modelView[2];
    invViewMatrix[9] = modelView[6];
    invViewMatrix[10] = modelView[10];
    invViewMatrix[11] = modelView[14];

    render();

    // display results
    glClear(GL_COLOR_BUFFER_BIT);

    // draw image from PBO
    glDisable(GL_DEPTH_TEST);

    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
#if 0
    // draw using glDrawPixels (slower)
    glRasterPos2i(0, 0);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
    glDrawPixels(width, height, GL_RGBA, GL_UNSIGNED_BYTE, 0);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
#else
    // draw using texture

    // copy from pbo to texture
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, 0);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

    // draw textured quad
    glEnable(GL_TEXTURE_2D);
    glBegin(GL_QUADS);
    glTexCoord2f(0, 0);
    glVertex2f(0, 0);
    glTexCoord2f(1, 0);
    glVertex2f(1, 0);
    glTexCoord2f(1, 1);
    glVertex2f(1, 1);
    glTexCoord2f(0, 1);
    glVertex2f(0, 1);
    glEnd();

    glDisable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, 0);
#endif

	//// Draw tweak bars
	//TwDraw();

    glutSwapBuffers();
    glutReportErrors();

    sdkStopTimer(&timer);

    computeFPS();
}

void idle()
{
    glutPostRedisplay();
}

inline void print_info()
{
	printf("density = %.2f, brightness = %.2f, transferOffset = %.2f, transferScale = %.2f\n", density, brightness, transferOffset, transferScale);
}

extern "C" void save_view(const char *file)
{
	printf("save view to %s\n", file);
	int regionSize = get_region_size();
	std::ofstream os(file);
	cereal::XMLOutputArchive archive(os);
	//archive(CEREAL_NVP(viewRotation.x), CEREAL_NVP(viewRotation.y), CEREAL_NVP(viewRotation.z), CEREAL_NVP(viewTranslation.x), CEREAL_NVP(viewTranslation.y), CEREAL_NVP(viewTranslation.z), CEREAL_NVP(loc.x), CEREAL_NVP(loc.y), CEREAL_NVP(regionSize));
	archive(CEREAL_NVP(viewRotation), CEREAL_NVP(viewTranslation), CEREAL_NVP(loc), CEREAL_NVP(regionSize));
}

extern "C" void load_view(const char *file)
{
	std::ifstream is(file);
	if (is.is_open())
	{
		printf("load view from %s\n", file);
		int regionSize = get_region_size();
		cereal::XMLInputArchive archive(is);
		//archive(CEREAL_NVP(viewRotation.x), CEREAL_NVP(viewRotation.y), CEREAL_NVP(viewRotation.z), CEREAL_NVP(viewTranslation.x), CEREAL_NVP(viewTranslation.y), CEREAL_NVP(viewTranslation.z), CEREAL_NVP(loc.x), CEREAL_NVP(loc.y), CEREAL_NVP(regionSize));
		archive(CEREAL_NVP(viewRotation), CEREAL_NVP(viewTranslation), CEREAL_NVP(loc), CEREAL_NVP(regionSize));
		locf.x = loc.x;
		locf.y = loc.y;
		loc2.x = loc.x;
		loc2.y = loc.y;
		set_region_size(regionSize);
	}
	else
	{
		printf("cannot open %s\n", file);
	}
}

/**
https://stackoverflow.com/questions/43624006/opengl-glreadpixels-to-freeimage-bitmap
https://stackoverflow.com/questions/33757634/creating-screenshot-in-opengl-not-working
*/
inline void save_rendering_to_image()
{
	GLubyte *pixels = (GLubyte*)malloc(width * height * 3);
	
	glPixelStorei(GL_PACK_ALIGNMENT, 1);
	glReadPixels(0, 0, width, height, GL_BGR, GL_UNSIGNED_BYTE, pixels);

	if (GL_NO_ERROR != glGetError())
	{
		std::cout<<"Error: Unable to read pixels."<<std::endl;
	}
	
	FIBITMAP* image = FreeImage_ConvertFromRawBits(pixels, width, height, 3 * width, 24, FI_RGBA_RED_MASK, FI_RGBA_GREEN_MASK, FI_RGBA_BLUE_MASK, false);

	char str[_MAX_PATH];
	sprintf(str, "~screenshot_%d.png", increase_screenshot_id());

	//if (FreeImage_Save(FIF_PNG, image, str, 0))
	//{
	//	std::cout << "Successfully saved " << str << std::endl;
	//}
	//else
	//{
	//	std::cout << "Failed saving " << str << std::endl;
	//}

	if(!FreeImage_Save(FIF_PNG, image, str, 0))
	{
		std::cout << "Failed saving " << str << std::endl;
	}

	FreeImage_Unload(image);

	free(pixels);
}

extern "C" void update_screenshots_in_Qt()
{
	if (qt_window)
	{
		qt_window->update_screenshots_later();
	}
}

extern "C" void save_rendering_and_display_in_Qt()
{
	save_rendering_to_image();
	update_screenshots_in_Qt();
}

extern "C" const char * apply_kmeans_and_save_image(const char *filename, unsigned char *h_output, uint *d_output, int k = 32)
{
	af::array img = af::loadImage(filename, true) / 255; // [0-255]
	int w = img.dims(0), h = img.dims(1), c = img.dims(2);
	printf("w=%d\th=%d\tc=%d\n", w, h, c);

	af::array vec = af::moddims(img, w * h, 1, c);
	af::array means_full, clusters_full;
	af::timer start1 = af::timer::start();
	kmeans(means_full, clusters_full, vec, k);
	auto t1 = af::timer::stop(start1);
	printf("K-means timing (seconds) \t k=%d time=%g \n", k, t1);
	af::array out_full = af::moddims(means_full(af::span, clusters_full, af::span), img.dims());
	sprintf(filename_buffer, "~%s", filename);
	af::saveImage(filename_buffer, out_full);
	printf("segmentation saved to %s\n", filename_buffer);

	//int color_components = c;
	//af::array h_A(w*h*color_components, h_output, afHost);
	//h_A = af::moddims(h_A / 255, w, h, color_components);
	//af::array d_A(w*h*color_components, d_output, afDevice);
	//d_A = af::moddims(d_A / 255, w, h, color_components);
	//char str[_MAX_PATH];
	//sprintf(str, "~host_%s", filename);
	//af::saveImage(str, h_A);
	//sprintf(str, "~device_%s", filename);
	//af::saveImage(str, d_A);
	//af::eval(out_full);
	//af::sync();

	return filename_buffer;
}

void keyboard(unsigned char key, int x, int y)
{
	//if (TwEventKeyboardGLUT(key, x, y))
	//{
	//	return;
	//}
	//printf("keyboard %d %d key %d \n", x, y, (int)key);
	auto c = get_SelectedColor();
	bool *p1 = get_ApplyAlpha(), *p2 = get_ApplyColor();
	clock_t start, end;
    switch (tolower(key))
    {
        case 27:
            #if defined (__APPLE__) || defined(MACOSX)
                exit(EXIT_SUCCESS);
            #else
                glutDestroyWindow(glutGetWindow());
                return;
            #endif
            break;

        case 'f':
            linearFiltering = !linearFiltering;
            setTextureFilterMode(linearFiltering);
            break;

        case '+':
            density += 0.01f;
			print_info();
            break;

        case '-':
            density -= 0.01f;
			print_info();
            break;

        case ']':
            brightness += 0.1f;
			print_info();
            break;

        case '[':
            brightness -= 0.1f;
			print_info();
            break;

        case ';':
            transferOffset += 0.01f;
			print_info();
            break;

        case '\'':
            transferOffset -= 0.01f;
			print_info();
            break;

        case '.':
            transferScale += 0.01f;
			print_info();
            break;

        case ',':
            transferScale -= 0.01f;
			print_info();
            break;

		case 'a':
			set_apply(true);
			if (qt_window)
			{
				qt_window->delay_draw_transfer_function_and_visibility_histograms();
			}
			break;

		case 'd':
			set_discard(true);
			if (qt_window)
			{
				qt_window->delay_draw_transfer_function_and_visibility_histograms();
			}
			break;

		case 's':
			set_save(true);
			break;

		case 'g':
			set_gaussian(true);
			if (qt_window)
			{
				qt_window->delay_draw_transfer_function_and_visibility_histograms();
			}
			break;

		case 'r':
			set_backup(true);
			break;

		case 'z':
			printf("enter color (e.g. 1 1 0) \n");
			float r, g, b;
			scanf("%g %g %g", &r, &g, &b);
			set_SelectedColor(r, g, b);
			printf("%g %g %g \n", (*c)[0], (*c)[1], (*c)[2]);
			break;

		case 'x':
			*p1 = !(*p1);
			printf("toggle alpha %s color %s \n", *p1 ? "true" : "false", *p2 ? "true" : "false");
			break;

		case 'c':
			*p2 = !(*p2);
			printf("toggle alpha %s color %s \n", *p1 ? "true" : "false", *p2 ? "true" : "false");
			break;

		case 'l':
			printf("enter loc (e.g. 307 194)\n");
			scanf("%d %d", &loc.x, &loc.y);
			printf("loc %d %d\n", loc.x, loc.y);
			break;

		case 'b':
			save_view(view_file);
			break;

		case 'v':
			load_view(view_file);
			break;

		case 'w':
			start = std::clock();
			//compute_saliency_once();
			compute_saliency();
			compute_feature_volume();
			compute_vws();
			compute_vws_array();
			end = std::clock();
			std::cout << "compute saliency, visibility and vws duration (seconds): " << (end - start) / (double)CLOCKS_PER_SEC << std::endl;
			break;

		case 'o':
			vws_tf_optimization();
			break;

		case 'p':
			vws_tf_optimization_linesearch();
			break;

		case 'i':
			add_volume_to_list_for_update_vortex();
			break;

		case 'j':
			add_volume_to_list_for_update_supernova();
			break;

		case 't':
			add_volume_to_list_for_update_test();
			break;

		case 'm':
			save_rendering_and_display_in_Qt();
			break;

		case 'n':
			set_save_ppm(true);
			break;

		case 'k':
			set_kmeans(true);
			break;

		case 'h':
			set_display_selection(false);
			break;

		case 'y':
			set_display_selection(true);
			break;

		case 'u':
			load_ppm_to_gpu();
			break;

		case 'e':
			erase_ppm_on_gpu();
			break;

		default:
            break;
    }

    glutPostRedisplay();
}

int ox, oy;
int buttonState = 0;

inline int2 convert_mouse_position(int x, int y)
{
	//loc.x = x - n / 2;
	// put the tip of mouse cursor at the center of the selected region
	//loc.y = h - y - n * 4 / 3;

	int h = glutGet(GLUT_WINDOW_HEIGHT);
	int n = get_region_size();
	//std::cout << "convert_mouse_position " << x << " " << h - y << std::endl;
	int2 ans;
	ans.x = x - n / 3;
	ans.y = h - y - n * 2 / 3;
	return ans;
}

void mouse(int button, int state, int x, int y)
{
	//if (TwEventMouseButtonGLUT(button, state, x, y))
	//{
	//	return;
	//}
	auto pos = unproject_mouse_position(x, y);
	if (button == GLUT_LEFT_BUTTON)
	{
		loc = convert_mouse_position(x, y);
		//loc = unproject_mouse_position(x, y);
		locf.x = loc.x;
		locf.y = loc.y;
		loc2.x = loc.x;
		loc2.y = loc.y;

		if (state == GLUT_UP && glutGetModifiers() == GLUT_ACTIVE_SHIFT)
		{
			if (!volume_list2.empty())
			{
				std::cout << "volume_list2.size=" << volume_list2.size() << "\tloc.x=" << loc.x << "\tloc.y=" << loc.y << std::endl;
				add_volume_to_list_for_update_reverse();
			}
		}
	}
	if (button == GLUT_RIGHT_BUTTON)
	{
		loc2 = convert_mouse_position(x, y);
		if (state == GLUT_DOWN)
		{
			std::cout<<"loc="<<loc.x<<","<<loc.y<<"\t loc2="<<loc2.x<<"," <<loc2.y<<std::endl;
		}
	}

    if (state == GLUT_DOWN)
    {
        buttonState  |= 1<<button;
    }
    else if (state == GLUT_UP)
    {
        buttonState = 0;
    }

    ox = x;
    oy = y;
    glutPostRedisplay();

	if (1 == state)
	{
		printf("mouse %d %d button %d state %d \t loc.x=%d loc.y=%d \t pos.x=%d pos.y=%d \n", x, y, button, state, loc.x, loc.y, pos.x, pos.y);
	}
}

void passive_motion(int x, int y)
{
	//printf("passive_motion %d %d \n", x, y);
	//int h = glutGet(GLUT_WINDOW_HEIGHT);
	if (glutGetModifiers() == GLUT_ACTIVE_CTRL)
	{
		int n = get_region_size();
		//loc.x = x - n / 2;
		//loc.y = height - y - n;
		// put the tip of mouse cursor at the center of the selected region
		//loc.y = h - y - n * 4 / 3;
		//loc.x = x;
		//loc.y = h - y;
		loc = convert_mouse_position(x, y);
		locf.x = loc.x;
		locf.y = loc.y;

		if (abs(loc2.x-loc.x) + abs(loc2.y-loc.y) > n>>4)
		{
			if (qt_window && qt_window->is_save_renderings_checked())
			{
				apply_save_rendering();
			}
		}
		//loc2.x = loc.x;
		//loc2.y = loc.y;
		loc2 = loc;
		//std::cout << "loc2=" << &loc2 << " " << loc2.x << " " << loc2.y << "\tloc=" << &loc << " " << loc.x << " " << loc.y << std::endl;

		reset_transfer_function();
		apply_tf_editing();
		
		//std::cout << "qt_window is " << (qt_window ? "valid" : "null") << std::endl;
		if (qt_window)
		{
			qt_window->delay_draw_transfer_function_and_visibility_histograms();
		}
	}
}

void motion(int x, int y)
{
	//if (TwEventMouseMotionGLUT(x, y))
	//{
	//	return;
	//}
	//printf("motion %d %d \n", x, y);

    float dx, dy;
    dx = (float)(x - ox);
    dy = (float)(y - oy);

    if (buttonState == 4)
    {
        // right = zoom
        viewTranslation.z += dy / 100.0f;
    }
    else if (buttonState == 2)
    {
        // middle = translate
        viewTranslation.x += dx / 100.0f;
        viewTranslation.y -= dy / 100.0f;
    }
    else if (buttonState == 1)
    {
        // left = rotate
        viewRotation.x += dy / 5.0f;
        viewRotation.y += dx / 5.0f;
    }

    ox = x;
    oy = y;
    glutPostRedisplay();
}

extern "C" int iDivUp(int a, int b)
{
    return (a % b != 0) ? (a / b + 1) : (a / b);
}

void reshape(int w, int h)
{
	//width = w;
	//height = h;
	printf("reshape: pixel buffer size %d %d \t window size %d %d \n", width, height, w, h);
	initPixelBuffer();

    // calculate new grid size
    gridSize = dim3(iDivUp(width, blockSize.x), iDivUp(height, blockSize.y));

    glViewport(0, 0, w, h);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);

	//// Send the new window size to AntTweakBar
	//TwWindowSize(width, height);
}

void cleanup()
{
	//TwTerminate();

    sdkDeleteTimer(&timer);

    freeCudaBuffers();

    if (pbo)
    {
        cudaGraphicsUnregisterResource(cuda_pbo_resource);
        glDeleteBuffers(1, &pbo);
        glDeleteTextures(1, &tex);
    }
    // Calling cudaProfilerStop causes all profile data to be
    // flushed before the application exits
    checkCudaErrors(cudaProfilerStop());
}

void initGL(int *argc, char **argv)
{
    // initialize GLUT callback functions
    glutInit(argc, argv);
    glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE);
    glutInitWindowSize(width, height);
	glutInitWindowPosition(100, 100);
    glutCreateWindow("CUDA volume rendering");

    if (!isGLVersionSupported(2,0) ||
        !areGLExtensionsSupported("GL_ARB_pixel_buffer_object"))
    {
        printf("Required OpenGL extensions are missing.");
        exit(EXIT_SUCCESS);
    }
}

void initPixelBuffer()
{
    if (pbo)
    {
        // unregister this buffer object from CUDA C
        checkCudaErrors(cudaGraphicsUnregisterResource(cuda_pbo_resource));

        // delete old buffer
        glDeleteBuffers(1, &pbo);
        glDeleteTextures(1, &tex);
    }

    // create pixel buffer object for display
    glGenBuffers(1, &pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, width*height*sizeof(GLubyte)*4, 0, GL_STREAM_DRAW_ARB);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

    // register this buffer object with CUDA
    checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, pbo, cudaGraphicsMapFlagsWriteDiscard));

    // create texture for display
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glBindTexture(GL_TEXTURE_2D, 0);
}

// Load raw data from disk
void *loadRawFile(const char *filename, size_t size)
{
    FILE *fp = fopen(filename, "rb");

    if (!fp)
    {
        fprintf(stderr, "Error opening file '%s'\n", filename);
        return 0;
    }

    void *data = malloc(size);
    size_t read = fread(data, 1, size, fp);
    fclose(fp);

//#if defined(_MSC_VER_)
//    printf("Read '%s', %Iu bytes\n", filename, read);
//#else
//    printf("Read '%s', %zu bytes\n", filename, read);
//#endif

    return data;
}

//// General initialization call for CUDA Device
//int chooseCudaDevice(int argc, const char **argv, bool bUseOpenGL)
//{
//    int result = 0;
//
//    if (bUseOpenGL)
//    {
//        result = findCudaGLDevice(argc, argv);
//    }
//    else
//    {
//        result = findCudaDevice(argc, argv);
//    }
//
//    return result;
//}

void runSingleTest(const char *ref_file, const char *exec_path)
{
    bool bTestResult = true;

    uint *d_output;
    checkCudaErrors(cudaMalloc((void **)&d_output, width*height*sizeof(uint)));
    checkCudaErrors(cudaMemset(d_output, 0, width*height*sizeof(uint)));

    float modelView[16] =
    {
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 4.0f, 1.0f
    };

    invViewMatrix[0] = modelView[0];
    invViewMatrix[1] = modelView[4];
    invViewMatrix[2] = modelView[8];
    invViewMatrix[3] = modelView[12];
    invViewMatrix[4] = modelView[1];
    invViewMatrix[5] = modelView[5];
    invViewMatrix[6] = modelView[9];
    invViewMatrix[7] = modelView[13];
    invViewMatrix[8] = modelView[2];
    invViewMatrix[9] = modelView[6];
    invViewMatrix[10] = modelView[10];
    invViewMatrix[11] = modelView[14];

    // call CUDA kernel, writing results to PBO
    copyInvViewMatrix(invViewMatrix, sizeof(float4)*3);

    // Start timer 0 and process n loops on the GPU
    int nIter = 10;

    for (int i = -1; i < nIter; i++)
    {
        if (i == 0)
        {
            cudaDeviceSynchronize();
            sdkStartTimer(&timer);
        }

        render_kernel(gridSize, blockSize, d_output, width, height, density, brightness, transferOffset, transferScale, loc);
    }

    cudaDeviceSynchronize();
    sdkStopTimer(&timer);
    // Get elapsed time and throughput, then log to sample and master logs
    double dAvgTime = sdkGetTimerValue(&timer)/(nIter * 1000.0);
    printf("volumeRender, Throughput = %.4f MTexels/s, Time = %.5f s, Size = %u Texels, NumDevsUsed = %u, Workgroup = %u\n",
           (1.0e-6 * width * height)/dAvgTime, dAvgTime, (width * height), 1, blockSize.x * blockSize.y);


    getLastCudaError("Error: render_kernel() execution FAILED");
    checkCudaErrors(cudaDeviceSynchronize());

    unsigned char *h_output = (unsigned char *)malloc(width*height*4);
    checkCudaErrors(cudaMemcpy(h_output, d_output, width*height*4, cudaMemcpyDeviceToHost));

    sdkSavePPM4ub("volume.ppm", h_output, width, height);
    bTestResult = sdkComparePPM("volume.ppm", sdkFindFilePath(ref_file, exec_path), MAX_EPSILON_ERROR, THRESHOLD, true);

    cudaFree(d_output);
    free(h_output);
    cleanup();

    exit(bTestResult ? EXIT_SUCCESS : EXIT_FAILURE);
}

//// load Gaussian kernels from files
//void load_gaussians()
//{
//	float a;
//	std::ifstream g5("gaussian_5_5_5.txt");
//	int number = R1*R1*R1;
//	for (int i = 0; i < number; i++)
//	{
//		g5 >> a;
//		gaussian5[i] = a;
//		//std::cout << gaussian5[i] << std::ends;
//	}
//	//std::cout << std::endl;
//	g5.close();
//
//	std::ifstream g9("gaussian_9_9_9.txt");
//	number = R2*R2*R2;
//	for (int i = 0; i < number; i++)
//	{
//		g9 >> a;
//		gaussian9[i] = a;
//		//std::cout << gaussian9[i] << std::ends;
//	}
//	//std::cout << std::endl;
//	g9.close();
//}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int init_gl_main(int argc, char **argv)
{
    pArgc = &argc;
    pArgv = argv;

    char *ref_file = NULL;

#if defined(__linux__)
    setenv ("DISPLAY", ":0", 0);
#endif

    //start logs
    printf("%s Starting...\n\n", sSDKsample);

    if (checkCmdLineFlag(argc, (const char **)argv, "file"))
    {
        getCmdLineArgumentString(argc, (const char **)argv, "file", &ref_file);
        fpsLimit = frameCheckNumber;
    }

	if (ref_file)
	{
		findCudaDevice(argc, (const char **)argv);
	}
	else
	{
		// First initialize OpenGL context, so we can properly set the GL for CUDA.
		// This is necessary in order to achieve optimal performance with OpenGL/CUDA interop.
		initGL(&argc, argv);

		findCudaDevice(argc, (const char **)argv);
	}

    // parse arguments
    char *filename;

    if (getCmdLineArgumentString(argc, (const char **) argv, "volume", &filename))
    {
        volumeFilename = filename;
    }

    int n;

    if (checkCmdLineFlag(argc, (const char **) argv, "size"))
    {
        n = getCmdLineArgumentInt(argc, (const char **) argv, "size");
        volumeSize.width = volumeSize.height = volumeSize.depth = n;
    }

    if (checkCmdLineFlag(argc, (const char **) argv, "xsize"))
    {
        n = getCmdLineArgumentInt(argc, (const char **) argv, "xsize");
        volumeSize.width = n;
    }

    if (checkCmdLineFlag(argc, (const char **) argv, "ysize"))
    {
        n = getCmdLineArgumentInt(argc, (const char **) argv, "ysize");
        volumeSize.height = n;
    }

    if (checkCmdLineFlag(argc, (const char **) argv, "zsize"))
    {
        n= getCmdLineArgumentInt(argc, (const char **) argv, "zsize");
        volumeSize.depth = n;
    }

	set_volume_file(volumeFilename, strlen(volumeFilename));
	program_path = argv[0];

    // load volume data
    char *path = sdkFindFilePath(volumeFilename, argv[0]);
	printf("volume %s\n", path);
	
    if (path == 0)
    {
        printf("Error finding file '%s'\n", volumeFilename);
        exit(EXIT_FAILURE);
    }

    size_t size = volumeSize.width*volumeSize.height*volumeSize.depth*sizeof(VolumeType);
    void *h_volume = loadRawFile(path, size);

	// load transfer function
	auto tf_path = sdkFindFilePath(tfFile, argv[0]);
	printf("transfer function %s\n", tf_path);
	openTransferFunctionFromVoreenXML(tf_path);

    initCuda(h_volume, volumeSize);

    free(h_volume);

    sdkCreateTimer(&timer);

    printf("Press '+' and '-' to change density (0.01 increments)\n"
           "      ']' and '[' to change brightness\n"
           "      ';' and ''' to modify transfer function offset\n"
           "      '.' and ',' to modify transfer function scale\n\n");

    // calculate new grid size
    gridSize = dim3(iDivUp(width, blockSize.x), iDivUp(height, blockSize.y));

    if (ref_file)
    {
        runSingleTest(ref_file, argv[0]);
    }
    else
    {
        // This is the normal rendering path for VolumeRender
        glutDisplayFunc(display);
        glutKeyboardFunc(keyboard);
        glutMouseFunc(mouse);
		glutMotionFunc(motion);
		glutPassiveMotionFunc(passive_motion);
		glutReshapeFunc(reshape);
        glutIdleFunc(idle);
		//glutPassiveMotionFunc((GLUTmousemotionfun)TwEventMouseMotionGLUT);
		//glutSpecialFunc((GLUTspecialfun)TwEventSpecialGLUT);
		//TwGLUTModifiersFunc(glutGetModifiers);

        initPixelBuffer();

#if defined (__APPLE__) || defined(MACOSX)
        atexit(cleanup);
#else
        glutCloseFunc(cleanup);
#endif

		//// Initialize AntTweakBar
		//TwInit(TW_OPENGL, NULL);
		//// Create a tweak bar
		//auto bar = TwNewBar("Blend");
		//TwDefine("Blend size='140 84' position='0 8'");
		//TwDefine(" Blend iconified=true ");  // Blend is iconified
		//TwDefine(" TW_HELP visible=false ");  // help bar is hidden
		//TwAddVarRW(bar, "alpha", TW_TYPE_BOOL32, get_ApplyAlpha(), "");
		//TwAddVarRW(bar, "color", TW_TYPE_BOOL32, get_ApplyColor(), "");
		//TwAddVarRW(bar, "pick", TW_TYPE_COLOR3F, get_SelectedColor(), "");

		//af::Window wnd("Image Demo");
		//af::array img = af::loadImage("../QtCuda/~screenshot_0.ppm", true);
		//wnd.image(img);
		//wnd.show();

        glutMainLoop();
    }
	return 0;
}

int main(int argc, char *argv[])
{
	af::info();
	QApplication a(argc, argv);
	MainWindow w;
	w.show();

	w.set_pointers(get_SelectedColor(), get_ApplyAlpha(), get_ApplyColor(), &time_varying_tf_editing, &time_varying_tf_reset, &time_varying_vws_optimization, &temporal_visibility);
	qt_window = &w;

	init_gl_main(argc, argv);

	return a.exec();
}
