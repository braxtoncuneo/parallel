__global__ void embarrassingly_easy(int *input, int *output) {
    size_t index = blockDim.x * blockIdx.x + threadIdx.x;
    int value = input[index];
    for(int i=0; i<4; i++) {
        value = A(value);
        value = B(value);
    }
    output[index] = value;
}