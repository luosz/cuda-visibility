//#include "kernel.h"
#include <stdio.h>
#include <stdlib.h>
#ifdef _WIN32
#define WINDOWS_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif
#ifdef __APPLE__
#include <GLUT/glut.h>
#else
#include <GL/glew.h>
#include <GL/freeglut.h>
#endif
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

void launch_kernel(uchar4 *d_out, int w, int h, int2 pos);

#define W 600
#define H 600
#define DELTA 5 // pixel increment for arrow keys
#define TITLE_STRING "flashlight: distance image display app"
int2 loc = { W / 2, H / 2 };
bool dragMode = false; // mouse tracking mode

void keyboard(unsigned char key, int x, int y) {
	if (key == 'a') dragMode = !dragMode; //toggle tracking mode
	if (key == 27) exit(0);
	glutPostRedisplay();
}

void mouseMove(int x, int y) {
	if (dragMode) return;
	loc.x = x;
	loc.y = y;
	glutPostRedisplay();
}

void mouseDrag(int x, int y) {
	if (!dragMode) return;
	loc.x = x;
	loc.y = y;
	glutPostRedisplay();
}

void handleSpecialKeypress(int key, int x, int y) {
	if (key == GLUT_KEY_LEFT)  loc.x -= DELTA;
	if (key == GLUT_KEY_RIGHT) loc.x += DELTA;
	if (key == GLUT_KEY_UP)    loc.y -= DELTA;
	if (key == GLUT_KEY_DOWN)  loc.y += DELTA;
	glutPostRedisplay();
}

void printInstructions() {
	printf("flashlight interactions\n");
	printf("a: toggle mouse tracking mode\n");
	printf("arrow keys: move ref location\n");
	printf("esc: close graphics qt_window\n");
}

// texture and pixel objects
GLuint pbo = 0; // OpenGL pixel buffer object
GLuint tex = 0; // OpenGL texture object
struct cudaGraphicsResource *cuda_pbo_resource;

void render() {
  uchar4 *d_out = 0;
  cudaGraphicsMapResources(1, &cuda_pbo_resource, 0);
  cudaGraphicsResourceGetMappedPointer((void **)&d_out, NULL,
                                       cuda_pbo_resource);
  launch_kernel(d_out, W, H, loc);
  cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0);
}

void drawTexture() {
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, W, H, 0, GL_RGBA,
               GL_UNSIGNED_BYTE, NULL);
  glEnable(GL_TEXTURE_2D);
  glBegin(GL_QUADS);
  glTexCoord2f(0.0f, 0.0f); glVertex2f(0, 0);
  glTexCoord2f(0.0f, 1.0f); glVertex2f(0, H);
  glTexCoord2f(1.0f, 1.0f); glVertex2f(W, H);
  glTexCoord2f(1.0f, 0.0f); glVertex2f(W, 0);
  glEnd();
  glDisable(GL_TEXTURE_2D);
}

void display() {
  render();
  drawTexture();
  glutSwapBuffers();
}

void initGLUT(int *argc, char **argv) {
  glutInit(argc, argv);
  glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
  glutInitWindowSize(W, H);
  glutCreateWindow(TITLE_STRING);
#ifndef __APPLE__
  glewInit();
#endif
}

void initPixelBuffer() {
  glGenBuffers(1, &pbo);
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
  glBufferData(GL_PIXEL_UNPACK_BUFFER, 4 * W*H*sizeof(GLubyte), 0,
               GL_STREAM_DRAW);
  glGenTextures(1, &tex);
  glBindTexture(GL_TEXTURE_2D, tex);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, pbo,
  cudaGraphicsMapFlagsWriteDiscard);
}

void exitfunc() {
  if (pbo) {
    cudaGraphicsUnregisterResource(cuda_pbo_resource);
    glDeleteBuffers(1, &pbo);
    glDeleteTextures(1, &tex);
  }
}

int main(int argc, char** argv) {
  printInstructions();
  initGLUT(&argc, argv);
  gluOrtho2D(0, W, H, 0);
  glutKeyboardFunc(keyboard);
  glutSpecialFunc(handleSpecialKeypress);
  glutPassiveMotionFunc(mouseMove);
  glutMotionFunc(mouseDrag);
  glutDisplayFunc(display);
  initPixelBuffer();
  glutMainLoop();
  atexit(exitfunc);
  return 0;
}