#ifndef CONNECTED_LAYER_H
#define CONNECTED_LAYER_H

#include "activations.h"
#include "layer.h"
#include "network.h"

layer make_connected_layer(int batch, int inputs, int outputs, ACTIVATION activation, int batch_normalize, int adam);

void forward_connected_layer(layer l, network net);
void backward_connected_layer(layer l, network net);
void update_connected_layer(layer l, update_args a);
#ifdef THREAD
void forward_connected_layer_thread(netlayer* input);
#endif

#ifdef GPU
void forward_connected_layer_gpu(layer l, network net);
#ifdef THREAD
//stream apply connected
void forward_connected_layer_gpu_thread(netlayer* input, int id);
#endif
void backward_connected_layer_gpu(layer l, network net);
void update_connected_layer_gpu(layer l, update_args a);
void push_connected_layer(layer l);
void pull_connected_layer(layer l);
#endif

#endif

