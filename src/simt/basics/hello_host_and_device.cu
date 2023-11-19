#include <cstdio>

__host__ __device__ void hello_printer(int value) {
    printf("The argument is %d.\n",value);
}

__global__ void hello(int value) {
    hello_printer(value);
}

int main(int argc, char *argv[]) {
    int value = (argc>1) ? atoi(argv[1]) : 0;
    hello<<<1,1>>>(value);
    cudaDeviceSynchronize();
    hello_printer(value*10);
}