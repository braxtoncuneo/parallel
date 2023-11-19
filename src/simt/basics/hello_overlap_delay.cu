#include <cstdio>
#include <chrono>
#include <thread>

__global__ void hello() {
    printf("Hello from the GPU!\n");
}

int main() {
    hello<<<1,1>>>();
    std::this_thread::sleep_for(std::chrono::seconds(1));
    printf("Hello from the CPU, before the sync!\n");
    cudaDeviceSynchronize();
    printf("Hello from the CPU, after the sync!\n");
}