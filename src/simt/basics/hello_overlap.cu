#include <cstdio>

__global__ void hello() {
    printf("Hello from the GPU!\n");
}

int main() {
    hello<<<1,1>>>();
    printf("Hello from the CPU, before the sync!\n");
    cudaDeviceSynchronize();
    printf("Hello from the CPU, after the sync!\n");
}