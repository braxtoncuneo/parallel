#include <cstdio>

__global__ void hello() {
    printf("Hello from the GPU!\n");
}

int main() {
    hello<<<1,1>>>();
    cudaDeviceSynchronize();
}