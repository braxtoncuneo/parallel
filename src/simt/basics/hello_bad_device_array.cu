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

    int *array = new int[size];

    for(size_t i=0; i<size; i++){
        array[i] = i;
    }
    print_array(array,size);

    array_square<<<1,32>>>(array,size);
    auto_throw(cudaDeviceSynchronize());

    print_array(array,size);

    delete[] array;
    return 0;
}