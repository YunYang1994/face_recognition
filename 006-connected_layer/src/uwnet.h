// Include guards and C++ compatibility
#ifndef UWNET_H
#define UWNET_H
#include "image.h"
#include "matrix.h"
#ifdef __cplusplus
extern "C" {
#endif

// Layer and network definitions

// The kinds of layers our framework supports
typedef enum{CONNECTED} LAYER_TYPE;

// The kinds of activations our framework supports
typedef enum{LINEAR, LOGISTIC, RELU, LRELU, SOFTMAX} ACTIVATION;

typedef struct layer {
    matrix *in;
    matrix *out;
    matrix *delta;

    // Weights
    matrix w;
    matrix dw;

    // Biases
    matrix b;
    matrix db;

    ACTIVATION activation;
    LAYER_TYPE type;
    matrix  (*forward)  (struct layer, struct matrix);
    void   (*backward) (struct layer, struct matrix);
    void   (*update)   (struct layer, float rate, float momentum, float decay);
} layer;

layer make_connected_layer(int inputs, int outputs, ACTIVATION activation);

typedef struct {
    layer *layers;
    int n;
} net;

matrix forward_net(net m, matrix X);
void backward_net(net m);
void update_net(net m, float rate, float momentum, float decay);

typedef struct{
    matrix X;
    matrix y;
} data;
data random_batch(data d, int n);
data load_image_classification_data(char *images, char *label_file);
void free_data(data d);
void train_image_classifier(net m, data d, int batch, int iters, float rate, float momentum, float decay);
float accuracy_net(net m, data d);

char *fgetl(FILE *fp);

void forward_bias(matrix m, matrix b);
void backward_bias(matrix delta, matrix db);
void activate_matrix(matrix m, ACTIVATION a);
void gradient_matrix(matrix m, ACTIVATION a, matrix d);

#ifdef __cplusplus
}
#endif
#endif
