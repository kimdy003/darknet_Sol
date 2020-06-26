//gpuindex cuda.c
int gpu_index = 0;

#ifdef GPU

#include "cuda.h"
#include "utils.h"
#include "blas.h"
#include <assert.h>
#include <stdlib.h>
#include <time.h>

void cuda_set_device(int n)
{
    gpu_index = n;
    cudaError_t status = cudaSetDevice(n);
    check_error(status);
}

int cuda_get_device()
{
    int n = 0;
    cudaError_t status = cudaGetDevice(&n);
    check_error(status);
    return n;
}

void check_error(cudaError_t status)
{
    //cudaDeviceSynchronize();
    cudaError_t status2 = cudaGetLastError();
    if (status != cudaSuccess)
    {
        const char *s = cudaGetErrorString(status);
        char buffer[256];
        printf("CUDA Error: %s\n", s);
        assert(0);
        snprintf(buffer, 256, "CUDA Error: %s", s);
        error(buffer);
    }
    if (status2 != cudaSuccess)
    {
        const char *s = cudaGetErrorString(status);
        char buffer[256];
        printf("CUDA Error Prev: %s\n", s);
        assert(0);
        snprintf(buffer, 256, "CUDA Error Prev: %s", s);
        error(buffer);
    }
}

//2020 0311 doyoung
void check_error_line(cudaError_t status, int line)
{
    //cudaDeviceSynchronize();
    cudaError_t status2 = cudaGetLastError();
    if (status != cudaSuccess)
    {
        const char *s = cudaGetErrorString(status);
        char buffer[256];
        printf("CUDA Error: %s, LINE : %d\n", s, line);
        assert(0);
        snprintf(buffer, 256, "CUDA Error: %s", s);
        error(buffer);
    }
    if (status2 != cudaSuccess)
    {
        const char *s = cudaGetErrorString(status);
        char buffer[256];
        printf("CUDA Error Prev: %s, LINE : %d\n", s, line);
        assert(0);
        snprintf(buffer, 256, "CUDA Error Prev: %s, LINE : %d\n", s, line);
        error(buffer);
    }
}

dim3 cuda_gridsize(size_t n)
{
    size_t k = (n - 1) / BLOCK + 1;
    size_t x = k;
    size_t y = 1;
    if (x > 65535)
    {
        x = ceil(sqrt(k));
        y = (n - 1) / (x * BLOCK) + 1;
    }
    dim3 d = {x, y, 1};
    //printf("%ld %ld %ld %ld\n", n, x, y, x*y*BLOCK);
    return d;
}
//int n_a = (int)n_des + (int)n_res + (int)n_vgg + (int)n_alex;

#ifdef CUDNN
    #ifdef THREAD
        #ifdef STREAM
            static cudaStream_t stream[THREAD_NUM_POOL];
            static cudnnHandle_t handle[THREAD_NUM_POOL];
            static int init_stream[THREAD_NUM_POOL] = {0};

            void cudnn_handle_set_stream()
            {
                for (int i = 0; i < THREAD_NUM_POOL; i++)
                {
                    cudnnCreate(&handle[i]);
                    cudaStreamCreate(&(stream[i]));
                    cudaError_t status = cudnnSetStream(handle[i], stream[i]);
                    check_error(status);
                    init_stream[i] = 1;
                }
            }

            //2020 0311 doyoung
            cudnnHandle_t cudnn_handle(int id, int line)
            {
                int i = id;
                if (!init_stream[i])
                {
                    cudnnCreate(&handle[i]);
                    cudaStreamCreate(&(stream[i]));
                    cudaError_t status = cudnnSetStream(handle[i], stream[i]);
                    check_error_line(status, line);
                    init_stream[i] = 1;
                }

                return handle[i];
            }

            void cuda_synchronize(int id, int line)
            {
                cudaError_t status = cudaStreamSynchronize(stream[id]);
                check_error_line(status, line);
            }
            cudaStream_t usedstream(int id)
            {
                return stream[id];
            }

        #else

            static int init_serial[n_a] = {0};
            static cudnnHandle_t handle[n_a];

            void cudnn_handle_set()
            {
                for (int i = 0; i < n_a; i++)
                {
                    cudnnCreate(&handle[i]);
                    init_serial[i] = 1;
                }
            }
            cudnnHandle_t cudnn_handle(int id, int line)
            {
                int i = id;
                if (!init_serial[i])
                {
                    cudnnCreate(&handle[i]);
                    init_serial[i] = 1;
                }
                return handle[i];
            }
        #endif
    #else
    //cudnn = 1, tread = 0, stream= 0
    static int init_a[n_a] = {0};
    static cudnnHandle_t handle[n_a];
    cudnnHandle_t cudnn_handle_a(int idx)
    {
        int i = idx;
        if (!init_a[i])
        {
            cudnnCreate(&handle[i]);
            init_a[i] = 1;
        }
        return handle[i];
    }
    cudnnHandle_t cudnn_handle()
    {
        int i = cuda_get_device();
        if (!init_a[i])
        {
            cudnnCreate(&handle[i]);
            init_a[i] = 1;
        }
        return handle[i];
    }
    #endif
#endif
static int init_blas[n_a] = {0};
static cublasHandle_t handle_blas[n_a];

cublasHandle_t blas_handle_a(int idx)
{
    int i = idx;
    if (!init_blas[i])
    {
        cublasCreate(&handle_blas[i]);
        init_blas[i] = 1;
    }

    return handle_blas[i];
}

cublasHandle_t blas_handle()
{
    int i = cuda_get_device();
    if (!init_blas[i])
    {
        cublasCreate(&handle_blas[i]);
        init_blas[i] = 1;
    }

    return handle_blas[i];
}

float *cuda_make_array(float *x, size_t n)
{
    float *x_gpu;
    size_t size = sizeof(float) * n;
    cudaError_t status = cudaMalloc((void **)&x_gpu, size);
#ifdef STREAM
    //2020 0311 doyoung
    cudaMemset(x_gpu, .0, size);
#endif

    check_error(status);
    if (x)
    {
        status = cudaMemcpy(x_gpu, x, size, cudaMemcpyHostToDevice);
        check_error(status);
    }
    else
    {
        fill_gpu(n, 0, x_gpu, 1);
    }
    if (!x_gpu)
        error("Cuda malloc failed\n");
    return x_gpu;
}

void cuda_random(float *x_gpu, size_t n)
{
    static curandGenerator_t gen[16];
    static int init[16] = {0};
    int i = cuda_get_device();
    if (!init[i])
    {
        curandCreateGenerator(&gen[i], CURAND_RNG_PSEUDO_DEFAULT);
        curandSetPseudoRandomGeneratorSeed(gen[i], time(0));
        init[i] = 1;
    }
    curandGenerateUniform(gen[i], x_gpu, n);
    check_error(cudaPeekAtLastError());
}

float cuda_compare(float *x_gpu, float *x, size_t n, char *s)
{
    float *tmp = calloc(n, sizeof(float));
    cuda_pull_array(x_gpu, tmp, n);
    //int i;
    //for(i = 0; i < n; ++i) printf("%f %f\n", tmp[i], x[i]);
    axpy_cpu(n, -1, x, 1, tmp, 1);
    float err = dot_cpu(n, tmp, 1, tmp, 1);
    printf("Error %s: %f\n", s, sqrt(err / n));
    free(tmp);
    return err;
}

int *cuda_make_int_array(int *x, size_t n)
{
    int *x_gpu;
    size_t size = sizeof(int) * n;
    cudaError_t status = cudaMalloc((void **)&x_gpu, size);
    check_error(status);
    if (x)
    {
        status = cudaMemcpy(x_gpu, x, size, cudaMemcpyHostToDevice);
        check_error(status);
    }
    if (!x_gpu)
        error("Cuda malloc failed\n");
    return x_gpu;
}

void cuda_free(float *x_gpu)
{
    cudaError_t status = cudaFree(x_gpu);
    check_error(status);
}

void cuda_freehost(float *x_gpu)
{
    cudaError_t status = cudaFreeHost(x_gpu);
    check_error(status);
}

void cuda_push_array(float *x_gpu, float *x, size_t n)
{
    size_t size = sizeof(float) * n;
    cudaError_t status = cudaMemcpy(x_gpu, x, size, cudaMemcpyHostToDevice);
    check_error(status);
}
#ifdef STREAM
//2020 0311 doyoung
    void cuda_pull_array_stream(float *x_gpu, float *x, size_t n, int id)
    {
        size_t size = sizeof(float) * n;
        cudaError_t status = cudaMemcpyAsync(x, x_gpu, size, cudaMemcpyDeviceToHost, stream[id]);
        check_error(status);
    }

    void cuda_push_array_stream(float *x_gpu, float *x, size_t n, int id)
    {
        size_t size = sizeof(float) * n;
        cudaError_t status = cudaMemcpyAsync(x_gpu, x, size, cudaMemcpyHostToDevice, stream[id]);
        check_error(status);
    }
#endif
void cuda_pull_array(float *x_gpu, float *x, size_t n)
{
    size_t size = sizeof(float) * n;
    cudaError_t status = cudaMemcpy(x, x_gpu, size, cudaMemcpyDeviceToHost);
    check_error(status);
}

float cuda_mag_array(float *x_gpu, size_t n)
{
    float *temp = calloc(n, sizeof(float));
    cuda_pull_array(x_gpu, temp, n);
    float m = mag_array(temp, n);
    free(temp);
    return m;
}

//2020 0311 doyoung
void cuda_malloc_int_host(int *x_host, size_t size, int line)
{
    cudaError_t status = cudaMallocHost((void **)&x_host, size);
    check_error_line(status, line);
}

void cuda_malloc_float_host(float **x_host, size_t size, int line)
{
    cudaError_t status = cudaMallocHost((void **)x_host, size);
    check_error_line(status, line);
}

#else
void cuda_set_device(int n)
{
}

#endif
