#include <cstdio>

__host__ __device__ void hello_printer(int value) {
    printf("The argument is %d.\n",value);
}

int main(int argc, char *argv[]) {
    int value = (argc>1) ? atoi(argv[1]) : 0;
    hello_printer(value);
    cudaDeviceSynchronize();
}