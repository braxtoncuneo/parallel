#include <cstdio>

__global__ void hello() {
    printf("Hello from thread %d!\n",threadIdx.x);
}

int main() {
    hello<<<1,4>>>();
    cudaDeviceSynchronize();
}