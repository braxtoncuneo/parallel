__global__ void embarrassingly_bad(int *input, int *output) {
    size_t index = blockDim.x * blockIdx.x + threadIdx.x;
    int value = input[index];
    for(int i=0; i<4; i++) {
        if((threadIdx.x+i)%2 == 0){
            value = A(value);
        } else {
            value = B(value);
        }
    }
    output[index] = value;
}