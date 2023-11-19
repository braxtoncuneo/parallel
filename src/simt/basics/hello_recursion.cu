#include <cstdio>
#include <cstdint>

__device__ uint64_t fibbonacci(uint64_t a, uint64_t b, uint64_t index) {
    if(index <= 0){
        return a;
    } else {
        return fibbonacci(b,a+b,index-1);
    }
}

__global__ void hello(uint64_t index) {
    printf("The %ldth fibbonacci number is %ld\n",index,fibbonacci(0,1,index));
}

int main(int argc, char *argv[]) {
    uint64_t index = (argc>1) ? atoi(argv[1]) : 0;
    hello<<<1,1>>>(index);
    cudaDeviceSynchronize();
}