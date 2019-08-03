// You probably don't want to edit this file
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#include "image.h"

image make_empty_image(int w, int h, int c)
{
    image out;
    out.data = 0;
    out.h = h;
    out.w = w;
    out.c = c;
    return out;
}

image make_image(int w, int h, int c)
{
    image out = make_empty_image(w,h,c);
    out.data = calloc(h*w*c, sizeof(float));
    return out;
}

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

void save_image_options(image im, const char *name, IMAGE_TYPE f, int quality)
{
    char buff[256];
    //sprintf(buff, "%s (%d)", name, windows);
    if(f == PNG)       sprintf(buff, "%s.png", name);
    else if (f == BMP) sprintf(buff, "%s.bmp", name);
    else if (f == TGA) sprintf(buff, "%s.tga", name);
    else if (f == JPG) sprintf(buff, "%s.jpg", name);
    else               sprintf(buff, "%s.png", name);
    unsigned char *data = calloc(im.w*im.h*im.c, sizeof(char));
    int i,k;
    for(k = 0; k < im.c; ++k){
        for(i = 0; i < im.w*im.h; ++i){
            data[i*im.c+k] = (unsigned char) (255*im.data[i + k*im.w*im.h]);
        }
    }
    int success = 0;
    if(f == PNG)       success = stbi_write_png(buff, im.w, im.h, im.c, data, im.w*im.c);
    else if (f == BMP) success = stbi_write_bmp(buff, im.w, im.h, im.c, data);
    else if (f == TGA) success = stbi_write_tga(buff, im.w, im.h, im.c, data);
    else if (f == JPG) success = stbi_write_jpg(buff, im.w, im.h, im.c, data, quality);
    free(data);
    if(!success) fprintf(stderr, "Failed to write image %s\n", buff);
}

void save_image(image im, const char *name)
{
    save_image_options(im, name, JPG, 80);
}

// 
// Load an image using stb
// channels = [0..4]
// channels > 0 forces the image to have that many channels
//
image load_image_stb(char *filename, int channels)
{
    int w, h, c;
    unsigned char *data = stbi_load(filename, &w, &h, &c, channels);
    if (!data) {
        fprintf(stderr, "Cannot load image \"%s\"\nSTB Reason: %s\n",
            filename, stbi_failure_reason());
        exit(0);
    }
    if (channels) c = channels;
    int i,j,k;
    image im = make_image(w, h, c);
    for(k = 0; k < c; ++k){
        for(j = 0; j < h; ++j){
            for(i = 0; i < w; ++i){
                int dst_index = i + w*j + w*h*k;
                int src_index = k + c*i + c*w*j;
                im.data[dst_index] = (float)data[src_index]/255.;
            }
        }
    }
    //We don't like alpha channels, #YOLO
    if(im.c == 4) im.c = 3;
    free(data);
    return im;
}

image load_image(char *filename)
{
    image out = load_image_stb(filename, 0);
    return out;
}

void free_image(image im)
{
    free(im.data);
}

float nn_interpolate(image im, float x, float y, int c)
{
    int lx = (int) round(x);
    int ly = (int) round(y);
    float v00 = get_pixel(im, lx, ly, c);
    return v00;
}

float bilinear_interpolate(image im, float x, float y, int c)
{
    int lx = (int) floor(x);
    int ly = (int) floor(y);
    float dx = x - lx;
    float dy = y - ly;
    float v00 = get_pixel(im, lx, ly, c);
    float v10 = get_pixel(im, lx+1, ly, c);
    float v01 = get_pixel(im, lx, ly+1, c);
    float v11 = get_pixel(im, lx+1, ly+1, c);
    float v =   v00*(1-dx)*(1-dy) + v10*dx*(1-dy) + 
                v01*(1-dx)*dy + v11*dx*dy;
    return v;
}

image bilinear_resize(image im, int w, int h)
{
    image r = make_image(w, h, im.c);   
    float xscale = (float)im.w/w;
    float yscale = (float)im.h/h;
    int i, j, k;
    for(k = 0; k < im.c; ++k){
        for(j = 0; j < h; ++j){
            for(i = 0; i < w; ++i){
                float y = (j+.5)*yscale - .5;
                float x = (i+.5)*xscale - .5;
                float val = bilinear_interpolate(im, x, y, k);
                set_pixel(r, i, j, k, val);
            }
        }
    }
    return r;
}

image nn_resize(image im, int w, int h)
{
    image r = make_image(w, h, im.c);   
    float xscale = (float)im.w/w;
    float yscale = (float)im.h/h;
    int i, j, k;
    for(k = 0; k < im.c; ++k){
        for(j = 0; j < h; ++j){
            for(i = 0; i < w; ++i){
                float y = (j+.5)*yscale - .5;
                float x = (i+.5)*xscale - .5;
                float val = nn_interpolate(im, x, y, k);
                set_pixel(r, i, j, k, val);
            }
        }
    }
    return r;
}

float get_pixel(image im, int x, int y, int c)
{
/*
    if(x >= im.w) return 0;
    if(y >= im.h) return 0;
    if(x < 0) return 0;
    if(y < 0) return 0;
    */

    if(x >= im.w) x = im.w - 1;
    if(y >= im.h) y = im.h - 1;
    if(x < 0) x = 0;
    if(y < 0) y = 0;
    assert(c >= 0);
    assert(c < im.c);
    return im.data[x + im.w*(y + im.h*c)];
}

void set_pixel(image im, int x, int y, int c, float v)
{
    assert(c >= 0);
    assert(c < im.c);
    if(x >= 0 && x < im.w && y >= 0 && y < im.h){
        im.data[x + im.w*(y + im.h*c)] = v;
    }
}

image copy_image(image im)
{
    image copy = make_image(im.w, im.h, im.c);
    memcpy(copy.data, im.data, im.w*im.h*im.c*sizeof(float));
    return copy;
}

image rgb_to_grayscale(image im)
{
    assert(im.c == 3);
    int i, j, k;
    image gray = make_image(im.w, im.h, 1);
    float scale[] = {0.299, 0.587, 0.114};
    for(k = 0; k < im.c; ++k){
        for(j = 0; j < im.h; ++j){
            for(i = 0; i < im.w; ++i){
                gray.data[i+im.w*j] += scale[k]*im.data[i + im.w*(j + im.h*k)];
            }
        }
    }
    return gray;
}

void scale_image(image im, int c, float v)
{
    int i;
    for(i = 0; i < im.w*im.h; ++i){
        im.data[i + c*im.w*im.h] *= v;
    }
}

void shift_image(image im, int c, float v)
{
    int i;
    for(i = 0; i < im.w*im.h; ++i){
        im.data[i + c*im.w*im.h] += v;
    }
}

void clamp_image(image im)
{
    int i;
    for(i = 0; i < im.w*im.h*im.c; ++i){
        im.data[i] = (im.data[i] < 0) ? 0 : ((im.data[i] > 1) ? 1 : im.data[i]);
    }
}

image grayscale_to_rgb(image im, float r, float g, float b)
{
    float color[3] = {r, g, b};
    assert(im.c == 1);
    int i, j, k;
    image rgb = make_image(im.w, im.h, 3);
    for(k = 0; k < 3; ++k){
        for(j = 0; j < im.h; ++j){
            for(i = 0; i < im.w; ++i){
                rgb.data[i + im.w*(j + im.h*k)] += color[k]*im.data[i+im.w*j];
            }
        }
    }
    return rgb;
}

image get_channel(image im, int c)
{
    int i,j;
    assert(c >= 0 && c < im.c);
    image chan = make_image(im.w, im.h, 1);
    for(j = 0; j < im.h; ++j){
        for(i = 0; i < im.w; ++i){
            chan.data[i+im.w*j] += im.data[i + im.w*(j + im.h*c)];
        }
    }
    return chan;
}

float three_way_max(float a, float b, float c)
{
    return (a > b) ? ( (a > c) ? a : c) : ( (b > c) ? b : c) ;
}

float three_way_min(float a, float b, float c)
{
    return (a < b) ? ( (a < c) ? a : c) : ( (b < c) ? b : c) ;
}

void rgb_to_hsv(image im)
{
    assert(im.c == 3);
    int i, j;
    float r, g, b;
    float h, s, v;
    for(j = 0; j < im.h; ++j){
        for(i = 0; i < im.w; ++i){
            r = get_pixel(im, i , j, 0);
            g = get_pixel(im, i , j, 1);
            b = get_pixel(im, i , j, 2);
            float max = three_way_max(r,g,b);
            float min = three_way_min(r,g,b);
            float delta = max - min;
            v = max;
            if(max == 0 || delta == 0){
                s = 0;
                h = 0;
            }else{
                s = delta/max;
                if(r == max){
                    h = (g - b) / delta;
                } else if (g == max) {
                    h = 2 + (b - r) / delta;
                } else {
                    h = 4 + (r - g) / delta;
                }
                if (h < 0) h += 6;
                h = h/6.;
            }
            set_pixel(im, i, j, 0, h);
            set_pixel(im, i, j, 1, s);
            set_pixel(im, i, j, 2, v);
        }
    }
}

void hsv_to_rgb(image im)
{
    assert(im.c == 3);
    int i, j;
    float r, g, b;
    float h, s, v;
    float f, p, q, t;
    for(j = 0; j < im.h; ++j){
        for(i = 0; i < im.w; ++i){
            h = 6 * get_pixel(im, i , j, 0);
            s = get_pixel(im, i , j, 1);
            v = get_pixel(im, i , j, 2);
            if (s == 0) {
                r = g = b = v;
            } else {
                int index = floor(h);
                f = h - index;
                p = v*(1-s);
                q = v*(1-s*f);
                t = v*(1-s*(1-f));
                if(index == 0){
                    r = v; g = t; b = p;
                } else if(index == 1){
                    r = q; g = v; b = p;
                } else if(index == 2){
                    r = p; g = v; b = t;
                } else if(index == 3){
                    r = p; g = q; b = v;
                } else if(index == 4){
                    r = t; g = p; b = v;
                } else {
                    r = v; g = p; b = q;
                }
            }
            set_pixel(im, i, j, 0, r);
            set_pixel(im, i, j, 1, g);
            set_pixel(im, i, j, 2, b);
        }
    }
}

image add_image(image a, image b)
{
    assert(a.w == b.w && a.h == b.h && a.c == b.c);
    int i;
    image sum = make_image(a.w, a.h, a.c);
    for(i = 0; i < a.w*a.h*a.c; ++i){
        sum.data[i] = a.data[i] + b.data[i];
    }
    return sum;
}

image sub_image(image a, image b)
{
    assert(a.w == b.w && a.h == b.h && a.c == b.c);
    int i;
    image sum = make_image(a.w, a.h, a.c);
    for(i = 0; i < a.w*a.h*a.c; ++i){
        sum.data[i] = a.data[i] - b.data[i];
    }
    return sum;
}
