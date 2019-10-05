#include "uwnet.h"

matrix forward_net(net m, matrix X)
{
    int i;
    for (i = 0; i < m.n; ++i) {
        layer l = m.layers[i];
        X = l.forward(l, X);
    }
    return X;
}

void backward_net(net m)
{
    int i;
    for (i = m.n-1; i >= 0; --i) {
        layer l = m.layers[i];
        matrix delta = {0};
        if(i > 0) delta = m.layers[i-1].delta[0];
        l.backward(l, delta);
    }
}

void update_net(net m, float rate, float momentum, float decay)
{
    int i;
    for(i = 0; i < m.n; ++i){
        layer l = m.layers[i];
        l.update(l, rate, momentum, decay);
    }
}
