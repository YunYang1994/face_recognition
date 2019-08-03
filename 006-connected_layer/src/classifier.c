#include <math.h>
#include <stdlib.h>
#include <assert.h>
#include "uwnet.h"
#include "matrix.h"

int max_index(float *a, int n)
{
    if(n <= 0) return -1;
    int i;
    int max_i = 0;
    float max = a[0];
    for (i = 1; i < n; ++i) {
        if (a[i] > max){
            max = a[i];
            max_i = i;
        }
    }
    return max_i;
}

float accuracy_net(net m, data d)
{
    matrix p = forward_net(m, d.X);
    int i;
    int correct = 0;
    for (i = 0; i < d.y.rows; ++i) {
        if (max_index(d.y.data + i*d.y.cols, d.y.cols) == max_index(p.data + i*p.cols, p.cols)) ++correct;
    }
    return (float)correct / d.y.rows;
}

float cross_entropy_loss(matrix y, layer l)
{
    matrix preds = l.out[0];
    matrix delta = l.delta[0];
    assert(y.rows == preds.rows);
    assert(y.cols == preds.cols);
    int i;
    float sum = 0;
    for(i = 0; i < y.cols*y.rows; ++i){
        sum += -y.data[i]*log(preds.data[i]);
        delta.data[i] += y.data[i] - preds.data[i];
    }
    return sum/y.rows;
}

void train_image_classifier(net m, data d, int batch, int iters, float rate, float momentum, float decay)
{
    int e;
    for(e = 0; e < iters; ++e){
        data b = random_batch(d, batch);
        forward_net(m, b.X);
        float err = cross_entropy_loss(b.y, m.layers[m.n-1]);
        fprintf(stderr, "%06d: Loss: %f\n", e, err);
        backward_net(m);
        update_net(m, rate/batch, momentum, decay);
        free_data(b);
    }
}
