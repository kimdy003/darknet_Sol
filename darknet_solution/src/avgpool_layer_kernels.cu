#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"

extern "C" {
#include "avgpool_layer.h"
#include "cuda.h"
}

__global__ void forward_avgpool_layer_kernel(int n, int w, int h, int c, float *input, float *output)
{
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(id >= n) return;

    int k = id % c;
    id /= c;
    int b = id;

    int i;
    int out_index = (k + c*b);
    output[out_index] = 0;
    for(i = 0; i < w*h; ++i){
        int in_index = i + h*w*(k + b*c);
        output[out_index] += input[in_index];
    }
    output[out_index] /= w*h;
}

__global__ void backward_avgpool_layer_kernel(int n, int w, int h, int c, float *in_delta, float *out_delta)
{
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(id >= n) return;

    int k = id % c;
    id /= c;
    int b = id;

    int i;
    int out_index = (k + c*b);
    for(i = 0; i < w*h; ++i){
        int in_index = i + h*w*(k + b*c);
        in_delta[in_index] += out_delta[out_index] / (w*h);
    }
}

extern "C" void forward_avgpool_layer_gpu(avgpool_layer layer, network net)
{
    size_t n = layer.c*layer.batch;

    forward_avgpool_layer_kernel<<<cuda_gridsize(n), BLOCK>>>(n, layer.w, layer.h, layer.c, net.input_gpu, layer.output_gpu);
    check_error(cudaPeekAtLastError());
}
#ifdef THREAD
extern "C" void forward_avgpool_layer_gpu_thread(netlayer* input, int id)
{
    network net = input->net;
    layer layer = input->layer;

    size_t n = layer.c*layer.batch;

    #ifdef STREAM
        //stream apply avgpool
        //fprintf(stderr, "[%d] index, avgpool id parameter : [%d] \n", net.index_n,  id);
        forward_avgpool_layer_kernel<<<cuda_gridsize(n), BLOCK, 0, usedstream(id)>>>(n, layer.w, layer.h, layer.c, net.input_gpu, layer.output_gpu);
        cuda_synchronize(id, __LINE__);
    #else
        forward_avgpool_layer_kernel<<<cuda_gridsize(n), BLOCK>>>(n, layer.w, layer.h, layer.c, net.input_gpu, layer.output_gpu);
    #endif
    check_error(cudaPeekAtLastError());

}
#endif

extern "C" void backward_avgpool_layer_gpu(avgpool_layer layer, network net)
{
    size_t n = layer.c*layer.batch;

    backward_avgpool_layer_kernel<<<cuda_gridsize(n), BLOCK>>>(n, layer.w, layer.h, layer.c, net.delta_gpu, layer.delta_gpu);
    check_error(cudaPeekAtLastError());
}

