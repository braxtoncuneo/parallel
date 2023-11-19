#include <cstdio>
#include <cstdint>
#include <stdexcept>
#include <string>

void auto_throw(cudaError_t status) {
    if(status != cudaSuccess) {
        std::string message = "ERROR: '";
        message += cudaGetErrorString(status);
        message +="'\n";
        throw std::runtime_error(message);
    }
}

__device__ uint64_t fibbonacci(uint64_t index) {
    if(index <= 2){
        return 1;
    }
    uint64_t final_a = fibbonacci(index-2);
    uint64_t final_b = fibbonacci(index-1);
    return final_a + final_b;
}

__global__ void hello(uint64_t index) {
    printf("The %ldth fibbonacci number is %ld\n",index,fibbonacci(index));
}

int main(int argc, char *argv[]) {
    uint64_t index = (argc>1) ? atoi(argv[1]) : 0;
    hello<<<1,1>>>(index);
    auto_throw(cudaDeviceSynchronize());
}