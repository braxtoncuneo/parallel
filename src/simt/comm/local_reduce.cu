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

__global__ void reduce(int* in_array, int* out_array, size_t size) {

    size_t thread_id = blockDim.x * blockIdx.x + threadIdx.x;

    // The shared memory used to communicate values
    __shared__ int shared_array[BLOCK_SIZE];

    // Keep intermediate values in private memory, only using shared
    // memory if the value must be read by the next reduction pass
    int local_value = A(in_array[thread_id]);
    if((threadIdx.x&1) != 0){
        shared_array[threadIdx.x] = local_value;
    }

    // Loop log2(BLOCK_SIZE) times
    __syncthreads();
    for(size_t i=1; i<BLOCK_SIZE; i<<=1){
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
        out_array[blockIdx.x] = local_value;
    }
}


int main(int argc, char *argv[]) {
    size_t size       = (argc>1) ? atoi(argv[1]) : 0;
    size_t out_size   = (size+BLOCK_SIZE-1) / BLOCK_SIZE;

    int *in_array;
    int *out_array;
    auto_throw(cudaMallocManaged(&in_array, size    *sizeof(int)));
    auto_throw(cudaMallocManaged(&out_array,out_size*sizeof(int)));

    // Generate input
    int total = 0;
    for(size_t i=0; i<size; i++){
        int value = rand()%10;
        total += value * 2;
        in_array[i] = value;
    }
    printf("The input:\n");
    print_array(in_array,size);

    reduce<<<out_size,BLOCK_SIZE>>>(in_array,out_array,size);
    auto_throw(cudaDeviceSynchronize());

    printf("The output, after reducing each block-sized sub-array:\n");
    print_array(out_array,out_size);

    int out_total = 0;
    for(size_t i=0; i<out_size; i++){
        out_total += out_array[i];
    }

    if(out_total != total) {
        printf("Error: %d != %d ", out_total, total);
    }

    auto_throw(cudaFree( in_array));
    auto_throw(cudaFree(out_array));

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
