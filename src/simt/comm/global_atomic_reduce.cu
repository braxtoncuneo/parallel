#include <cstdio>
#include <iostream>

// Defined later, as shown previously
void print_array(int* array, size_t size);
void auto_throw(cudaError_t status);

// The map operation
__device__ int A(int input) {
    return input * 2;
}

// The reduction operation
__device__ int B(int left, int right) {
    return left + right;
}

const size_t GROUP_SIZE = 1024;

__global__ void reduce(int* in_array, int* out_value, size_t size) {

    size_t thread_id = blockDim.x * blockIdx.x + threadIdx.x;

    // The shared memory used to communicate values
    __shared__ int shared_array[GROUP_SIZE];

    // Keep intermediate values in private memory, only using shared
    // memory if the value must be read by the next reduction pass
    int local_value = A(in_array[thread_id]);
    if((threadIdx.x&1) != 0){
        shared_array[threadIdx.x] = local_value;
    }

    // Loop log2(GROUP_SIZE) times
    __syncthreads();
    for(size_t i=1; i<GROUP_SIZE; i<<=1){
        if((i&threadIdx.x) == 0){
            size_t other_index = threadIdx.x^i;
            local_value = B(local_value,shared_array[other_index]);
            if(((i<<1)&threadIdx.x) != 0) {
                shared_array[threadIdx.x] = local_value;
            }
        }
        __syncthreads();
    }

    if(threadIdx.x == 0){
        atomicAdd(out_value,local_value);
    }
}


int main(int argc, char *argv[]) {
    size_t size       = (argc>1) ? atoi(argv[1]) : 0;
    size_t out_size   = (size+GROUP_SIZE-1) / GROUP_SIZE;

    int *in_array;
    int *out_value;
    auto_throw(cudaMallocManaged(&in_array, sizeof(int)*size));
    auto_throw(cudaMallocManaged(&out_value,sizeof(int)));

    // Generate input
    for(size_t i=0; i<size; i++){
        int value = rand()%10;
        in_array[i] = value;
    }

    print_array(in_array,size);

    reduce<<<out_size,GROUP_SIZE>>>(in_array,out_value,size);
    auto_throw(cudaDeviceSynchronize());

    print_array(out_value,1);

    auto_throw(cudaFree( in_array));
    auto_throw(cudaFree(out_value));

    return 0;
}


void auto_throw(cudaError_t status) {
    if(status != cudaSuccess) {
        std::string message = "ERROR: '";
        message += cudaGetErrorString(status);
        message +="'\n";
        throw std::runtime_error(message);
    }
}

void print_array(int* array, size_t size) {
    for(size_t i=0; i<size; i++){
        if(i != 0){
            std::cout << ',';
        }
        std::cout << array[i];
    }
    std::cout << '\n';
}
