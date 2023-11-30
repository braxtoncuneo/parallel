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

const size_t BLOCK_SIZE = 1024;


__global__ void map(int* inout_array, size_t size) {
    size_t thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    inout_array[thread_id] = A(inout_array[thread_id]);
}

__global__ void reduce(int* inout_array, size_t size, size_t i) {

    size_t thread_id = blockDim.x * blockIdx.x + threadIdx.x;

    if((i&thread_id) == 0){
        size_t other_index = thread_id^i;
        inout_array[thread_id] = B(inout_array[thread_id],inout_array[other_index]);
    }
}


int main(int argc, char *argv[]) {
    size_t size       = (argc>1) ? atoi(argv[1]) : 0;
    size_t out_size   = (size+BLOCK_SIZE-1) / BLOCK_SIZE;

    int *inout_array;
    auto_throw(cudaMallocManaged(&inout_array, size    *sizeof(int)));

    // Generate input
    int total = 0;
    for(size_t i=0; i<size; i++){
        int value = rand()%10;
        total += value * 2;
        inout_array[i] = value;
    }
    printf("The input:\n");
    print_array(inout_array,size);

    map<<<out_size,BLOCK_SIZE>>>(inout_array,size);
    for(size_t i=1; i<size; i<<=1){
        reduce<<<out_size,BLOCK_SIZE>>>(inout_array,size,i);
    }
    auto_throw(cudaDeviceSynchronize());

    printf("The output, after reducing the entire array:\n");
    print_array(inout_array,1);

    int out_total = inout_array[0];

    if(out_total != total) {
        printf("Error: %d != %d ", out_total, total);
    }

    auto_throw(cudaFree(inout_array));

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
