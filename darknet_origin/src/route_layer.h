#ifndef ROUTE_LAYER_H
#define ROUTE_LAYER_H
#include "network.h"
#include "layer.h"

typedef layer route_layer;

route_layer make_route_layer(int batch, int n, int *input_layers, int *input_size);
void forward_route_layer(const route_layer l, network net);
void backward_route_layer(const route_layer l, network net);
void resize_route_layer(route_layer *l, network *net);

#ifdef THREAD
void forward_route_layer_thread(netlayer* input);
#endif

#ifdef GPU
void forward_route_layer_gpu(const route_layer l, network net);
#ifdef THREAD
//stream apply route
void forward_route_layer_gpu_thread(netlayer* input, int id);
#endif
void backward_route_layer_gpu(const route_layer l, network net);
#endif

#endif
