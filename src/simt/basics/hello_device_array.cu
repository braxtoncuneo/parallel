#include <cstdio>
#include <iostream>

void auto_throw(cudaError_t status) {
    if(status != cudaSuccess) {
        std::string message = "ERROR: '";
        message += cudaGetErrorString(status);
        message +="'\n";
        throw std::runtime_error(message);
    }
}

__global__ void array_square(int* array, size_t size) {
    for(size_t i=threadIdx.x; i<size; i+=32){
        array[i] = array[i] * array[i];
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

int main(int argc, char *argv[]) {
    size_t size = (argc>1) ? atoi(argv[1]) : 0;

    int *cpu_array = new int[size];
    int *gpu_array;
    auto_throw(cudaMalloc(&gpu_array,size*sizeof(int)));

    for(size_t i=0; i<size; i++){
        cpu_array[i] = i;
    }
    print_array(cpu_array,size);

    auto_throw(cudaMemcpy(
        gpu_array,
        cpu_array,
        size*sizeof(int),
        cudaMemcpyHostToDevice
    ));
    auto_throw(cudaDeviceSynchronize());

    array_square<<<1,32>>>(gpu_array,size);
    auto_throw(cudaDeviceSynchronize());

    auto_throw(cudaMemcpy(
        cpu_array,
        gpu_array,
        size*sizeof(int),
        cudaMemcpyDeviceToHost
    ));
    auto_throw(cudaDeviceSynchronize());

    print_array(cpu_array,size);

    auto_throw(cudaFree(gpu_array));
    delete[] cpu_array;
    return 0;
}