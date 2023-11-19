#include <cstdio>

__global__ void hello() {
    int global_x_size = gridDim.x * blockDim.x;
    int global_y_size = gridDim.y * blockDim.y;
    int x_global_index = blockIdx.x * blockDim.x + threadIdx.x;
    int y_global_index = blockIdx.y * blockDim.y + threadIdx.y;
    int z_global_index = blockIdx.z * blockDim.z + threadIdx.z;
    int id = 0;
    id = (id+z_global_index)*global_y_size;
    id = (id+y_global_index)*global_x_size;
    id = id+x_global_index;

    printf(
        "Hello from block (%d,%d,%d), thread (%d,%d,%d) aka thread %d!\n",
        blockIdx.x, blockIdx.y, blockIdx.z,
        threadIdx.x,threadIdx.y,threadIdx.z,
        id
    );
}

int main() {
    dim3 dims(2,2,2);
    hello<<<dims,dims>>>();
    cudaDeviceSynchronize();
}