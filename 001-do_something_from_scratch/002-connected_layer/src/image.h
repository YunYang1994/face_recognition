#ifndef IMAGE_H
#define IMAGE_H
#include <stdio.h>

// DO NOT CHANGE THIS FILE

#include "matrix.h"
#define TWOPI 6.2831853

#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))

typedef struct{
    int w,h,c;
    float *data;
} image;

// Basic operations
float get_pixel(image im, int x, int y, int c);
void set_pixel(image im, int x, int y, int c, float v);
image copy_image(image im);
image rgb_to_grayscale(image im);
image grayscale_to_rgb(image im, float r, float g, float b);
void rgb_to_hsv(image im);
void hsv_to_rgb(image im);
void shift_image(image im, int c, float v);
void scale_image(image im, int c, float v);
void clamp_image(image im);
image get_channel(image im, int c);
int same_image(image a, image b);
image sub_image(image a, image b);
image add_image(image a, image b);

// Loading and saving
typedef enum{ PNG, BMP, TGA, JPG } IMAGE_TYPE;
image make_image(int w, int h, int c);
image load_image(char *filename);
void save_image_options(image im, const char *name, IMAGE_TYPE f, int quality);
void save_image(image im, const char *name);
void free_image(image im);

// Resizing
float nn_interpolate(image im, float x, float y, int c);
image nn_resize(image im, int w, int h);
float bilinear_interpolate(image im, float x, float y, int c);
image bilinear_resize(image im, int w, int h);

// Used for saving images

#endif

