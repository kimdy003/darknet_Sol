#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"

extern "C" {
#include "dropout_layer.h"
#include "cuda.h"
#include "utils.h"
}

__global__ void yoloswag420blazeit360noscope(float *input, int size, float *rand, float prob, float scale)
{
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(id < size) input[id] = (rand[id] < prob) ? 0 : input[id]*scale;
}

void forward_dropout_layer_gpu(dropout_layer layer, network net)
{
    if (!net.train) return;
    int size = layer.inputs*layer.batch;
    cuda_random(layer.rand_gpu, size);
    /*
    int i;
    for(i = 0; i < size; ++i){
        layer.rand[i] = rand_uniform();
    }
    cuda_push_array(layer.rand_gpu, layer.rand, size);
    */

    yoloswag420blazeit360noscope<<<cuda_gridsize(size), BLOCK>>>(net.input_gpu, size, layer.rand_gpu, layer.probability, layer.scale);
    check_error(cudaPeekAtLastError());
}

#ifdef THREAD
void forward_dropout_layer_gpu_thread(netlayer* input, int id)
{
    network net = input->net;
    layer layer = input->layer;
    
    if (!net.train) return;
    int size = layer.inputs*layer.batch;
    cuda_random(layer.rand_gpu, size);
    #ifdef STREAM
        //stream apply dropout
    	fprintf(stderr, "[%d] index, drop id parameter : [%d] \n", net.index_n, id);
        yoloswag420blazeit360noscope<<<cuda_gridsize(size), BLOCK, 0, usedstream(id)>>>(net.input_gpu, size, layer.rand_gpu, layer.probability, layer.scale);
        cuda_synchronize(id, __LINE__);
    #else
        yoloswag420blazeit360noscope<<<cuda_gridsize(size), BLOCK>>>(net.input_gpu, size, layer.rand_gpu, layer.probability, layer.scale);
    #endif
    check_error(cudaPeekAtLastError());
     
}
#endif


void backward_dropout_layer_gpu(dropout_layer layer, network net)
{
    if(!net.delta_gpu) return;
    int size = layer.inputs*layer.batch;

    yoloswag420blazeit360noscope<<<cuda_gridsize(size), BLOCK>>>(net.delta_gpu, size, layer.rand_gpu, layer.probability, layer.scale);
    check_error(cudaPeekAtLastError());
}