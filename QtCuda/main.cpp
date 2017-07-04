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
#include <helper_cuda_gl.h>

// Helper functions
#include <helper_cuda.h>
#include <helper_functions.h>
#include <helper_timer.h>

#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <helper_math.h>
//#include <AntTweakBar.h>
#include "tinyxml2.h"
#include "util.h"
#include "def.h"

// include cereal for serialization
#include "cereal/archives/xml.hpp"

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

const char *tfs[] = { "nucleon_naive_proportional_2.tfi","vortex_naive_proportional_2.tfi","CT-Knee_spectrum_6.tfi","E_1324_Rainbow6_even_2.tfi" };
const char *volumes[] = { "nucleon.raw","vorts1.raw","CT-Knee.raw","E_1324.raw" };
const int data_index = 0;
const char *tfFile = tfs[data_index];
const char *volumeFilename = volumes[data_index];
/**
41, 41, 41
128, 128, 128
379, 229, 305
432, 432, 432
*/
cudaExtent volumeSize = make_cudaExtent(41, 41, 41);
typedef unsigned char VolumeType;

int2 loc = {0, 0};
bool dragMode = false; // mouse tracking mode

//uint width = 512, height = 512;
uint width = D_WIDTH, height = D_HEIGHT;
dim3 blockSize(16, 16);
dim3 gridSize;

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

typedef float(*Pointer)[4];
extern "C" Pointer get_SelectedColor();
extern "C" void set_SelectedColor(float r, float g, float b);
extern "C" bool* get_ApplyColor();
extern "C" bool* get_ApplyAlpha();
extern "C" int get_region_size();
extern "C" float4* get_tf_array();
extern "C" int get_bin_count();
extern "C" bool get_apply();
extern "C" void set_apply(bool value);
extern "C" bool get_save();
extern "C" void set_save(bool value);
extern "C" bool get_discard();
extern "C" void set_discard(bool value);
extern "C" bool get_gaussian();
extern "C" void set_gaussian(bool value);
extern "C" bool get_backup();
extern "C" void set_backup(bool value);
extern "C" void set_volume_file(const char *file, int n);
extern "C" void backup_tf();
extern "C" void restore_tf();

extern "C" void setTextureFilterMode(bool bLinearFilter);
extern "C" void initCuda(void *h_volume, cudaExtent volumeSize);
extern "C" void freeCudaBuffers();
extern "C" void render_kernel(dim3 gridSize, dim3 blockSize, uint *d_output, uint imageW, uint imageH,
                              float density, float brightness, float transferOffset, float transferScale, int2 loc);
extern "C" void copyInvViewMatrix(float *invViewMatrix, size_t sizeofMatrix);

void initPixelBuffer();

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

	backup_tf();
}

/// open Voreen transfer functions
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
	std::vector<float> intensity_list;
	std::vector<float4> rgba_list;
	//intensity_list_clear();
	//colour_list_clear();
	//opacity_list_clear();
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
	load_lookuptable(intensity_list, rgba_list);
}

void computeFPS()
{
    frameCount++;
    fpsCount++;

    if (fpsCount == fpsLimit)
    {
        char fps[256];
        float ifps = 1.f / (sdkGetAverageTimerValue(&timer) / 1000.f);
		//sprintf(fps, "Volume Render: %3.1f fps", ifps);
        sprintf(fps, "CUDA volume rendering: %3.1f fps", ifps);

        glutSetWindowTitle(fps);
        fpsCount = 0;

        fpsLimit = (int)MAX(1.f, ifps);
        sdkResetTimer(&timer);
    }
}

// render image using CUDA
void render()
{
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

void keyboard(unsigned char key, int x, int y)
{
	//if (TwEventKeyboardGLUT(key, x, y))
	//{
	//	return;
	//}
	printf("keyboard %d %d key %d \n", x, y, (int)key);
	auto c = get_SelectedColor();
	bool *p1 = get_ApplyAlpha(), *p2 = get_ApplyColor();
    switch (key)
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
			break;

		case 'd':
			set_discard(true);
			break;

		case 's':
			set_save(true);
			break;

		case 'g':
			set_gaussian(true);
			break;

		case 't':
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
			{
				printf("save view to %s\n", view_file);
				std::ofstream os(view_file);
				cereal::XMLOutputArchive archive(os);
				archive(viewRotation.x, viewRotation.y, viewRotation.z, viewTranslation.x, viewTranslation.y, viewTranslation.z, loc.x, loc.y);
			}
			break;

		case 'v':
			{
				std::ifstream is(view_file);
				if (is.is_open())
				{
					printf("load view from %s\n", view_file);
					cereal::XMLInputArchive archive(is);
					archive(viewRotation.x, viewRotation.y, viewRotation.z, viewTranslation.x, viewTranslation.y, viewTranslation.z, loc.x, loc.y);
				}
				else
				{
					printf("cannot open %s\n", view_file);
				}
			}
			break;

		default:
            break;
    }

    glutPostRedisplay();
}

int ox, oy;
int buttonState = 0;

void mouse(int button, int state, int x, int y)
{
	//if (TwEventMouseButtonGLUT(button, state, x, y))
	//{
	//	return;
	//}
	//if (1==state)
	//{
	//	printf("mouse %d %d button %d state %d \n", x, y, button, state);
	//}

	auto n = get_region_size();
	loc.x = x-n/2;
	loc.y = height-y-n;

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

int iDivUp(int a, int b)
{
    return (a % b != 0) ? (a / b + 1) : (a / b);
}

void reshape(int w, int h)
{
    width = w;
    height = h;
	printf("reshape %d %d \n", width, height);
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
void *loadRawFile(char *filename, size_t size)
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

#if defined(_MSC_VER_)
    printf("Read '%s', %Iu bytes\n", filename, read);
#else
    printf("Read '%s', %zu bytes\n", filename, read);
#endif

    return data;
}

// General initialization call for CUDA Device
int chooseCudaDevice(int argc, const char **argv, bool bUseOpenGL)
{
    int result = 0;

    if (bUseOpenGL)
    {
        result = findCudaGLDevice(argc, argv);
    }
    else
    {
        result = findCudaDevice(argc, argv);
    }

    return result;
}

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

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
gl_main(int argc, char **argv)
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
        // use command-line specified CUDA device, otherwise use device with highest Gflops/s
        chooseCudaDevice(argc, (const char **)argv, false);
    }
    else
    {
        // First initialize OpenGL context, so we can properly set the GL for CUDA.
        // This is necessary in order to achieve optimal performance with OpenGL/CUDA interop.
        initGL(&argc, argv);

        // use command-line specified CUDA device, otherwise use device with highest Gflops/s
        chooseCudaDevice(argc, (const char **)argv, true);
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

        glutMainLoop();
    }
}

int main(int argc, char *argv[])
{
	QApplication a(argc, argv);
	MainWindow w;
	w.show();

	w.set_color_pointer(get_SelectedColor(), get_ApplyAlpha(), get_ApplyColor());

	gl_main(argc, argv);

	return a.exec();
}
