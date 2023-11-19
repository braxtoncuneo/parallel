#include <cstdio>

__global__ void hello() {
    printf(
        "Hello from thread (%d,%d,%d)!\n",
        threadIdx.x,threadIdx.y,threadIdx.z
    );
}

int main() {
    dim3 block_dims(2,2,2);
    hello<<<1,block_dims>>>();
    cudaDeviceSynchronize();
}