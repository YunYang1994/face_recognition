import sys, os
from ctypes import *
import math
import random

lib = CDLL(os.path.join(os.path.dirname(__file__), "libuwnet.so"), RTLD_GLOBAL)

def c_array(ctype, values):
    arr = (ctype*len(values))()
    arr[:] = values
    return arr

class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]
    def __add__(self, other):
        return add_image(self, other)
    def __sub__(self, other):
        return sub_image(self, other)

class MATRIX(Structure):
    _fields_ = [("rows", c_int),
                ("cols", c_int),
                ("data", POINTER(POINTER(c_float))),
                ("shallow", c_int)]

class DATA(Structure):
    _fields_ = [("X", MATRIX),
                ("y", MATRIX)]

class LAYER(Structure):
    pass

LAYER._fields_ = [("in",  POINTER(MATRIX)),
                ("out",   POINTER(MATRIX)),
                ("delta", POINTER(MATRIX)),
                ("w", MATRIX),
                ("dw", MATRIX),
                ("b", MATRIX),
                ("db", MATRIX),
                ("activation", c_int),
                ("type", c_int),
                ("forward", CFUNCTYPE(MATRIX, POINTER(LAYER), MATRIX)),
                ("backward", CFUNCTYPE(None, POINTER(LAYER), MATRIX)),
                ("update", CFUNCTYPE(None, POINTER(LAYER), c_float, c_float, c_float))]

class NET(Structure):
    _fields_ = [("layers", POINTER(LAYER)),
                ("n", c_int)]


(LINEAR, LOGISTIC, RELU, LRELU, SOFTMAX) = range(5)


add_image = lib.add_image
add_image.argtypes = [IMAGE, IMAGE]
add_image.restype = IMAGE

sub_image = lib.sub_image
sub_image.argtypes = [IMAGE, IMAGE]
sub_image.restype = IMAGE

make_image = lib.make_image
make_image.argtypes = [c_int, c_int, c_int]
make_image.restype = IMAGE

free_image = lib.free_image
free_image.argtypes = [IMAGE]

get_pixel = lib.get_pixel
get_pixel.argtypes = [IMAGE, c_int, c_int, c_int]
get_pixel.restype = c_float

set_pixel = lib.set_pixel
set_pixel.argtypes = [IMAGE, c_int, c_int, c_int, c_float]
set_pixel.restype = None

rgb_to_grayscale = lib.rgb_to_grayscale
rgb_to_grayscale.argtypes = [IMAGE]
rgb_to_grayscale.restype = IMAGE

copy_image = lib.copy_image
copy_image.argtypes = [IMAGE]
copy_image.restype = IMAGE

rgb_to_hsv = lib.rgb_to_hsv
rgb_to_hsv.argtypes = [IMAGE]
rgb_to_hsv.restype = None

clamp_image = lib.clamp_image
clamp_image.argtypes = [IMAGE]
clamp_image.restype = None

hsv_to_rgb = lib.hsv_to_rgb
hsv_to_rgb.argtypes = [IMAGE]
hsv_to_rgb.restype = None

shift_image = lib.shift_image
shift_image.argtypes = [IMAGE, c_int, c_float]
shift_image.restype = None

load_image_lib = lib.load_image
load_image_lib.argtypes = [c_char_p]
load_image_lib.restype = IMAGE

def load_image(f):
    return load_image_lib(f.encode('ascii'))

# Filetypes
(PNG, BMP, TGA, JPG) = range(4)

save_image_options_lib = lib.save_image_options
save_image_options_lib.argtypes = [IMAGE, c_char_p, c_int, c_int]
save_image_options_lib.restype = None

def save_image(im, f):
    return save_image_options_lib(im, f.encode('ascii'), JPG, 80)

def save_png(im, f):
    return save_image_options_lib(im, f.encode('ascii'), PNG, 0)


nn_resize = lib.nn_resize
nn_resize.argtypes = [IMAGE, c_int, c_int]
nn_resize.restype = IMAGE

bilinear_resize = lib.bilinear_resize
bilinear_resize.argtypes = [IMAGE, c_int, c_int]
bilinear_resize.restype = IMAGE




train_image_classifier = lib.train_image_classifier
train_image_classifier.argtypes = [NET, DATA, c_int, c_int, c_float, c_float, c_float]
train_image_classifier.restype = None

accuracy_net = lib.accuracy_net
accuracy_net.argtypes = [NET, DATA]
accuracy_net.restype = c_float

forward_net = lib.forward_net
forward_net.argtypes = [NET, MATRIX]
forward_net.restype = MATRIX

load_image_classification_data = lib.load_image_classification_data
load_image_classification_data.argtypes = [c_char_p, c_char_p]
load_image_classification_data.restype = DATA

make_connected_layer = lib.make_connected_layer
make_connected_layer.argtypes = [c_int, c_int, c_int]
make_connected_layer.restype = LAYER

def make_net(layers):
    m = NET()
    m.n = len(layers)
    m.layers = (LAYER*m.n) (*layers)
    return m

if __name__ == "__main__":
    im = load_image("data/dog.jpg")
    save_image(im, "hey")

