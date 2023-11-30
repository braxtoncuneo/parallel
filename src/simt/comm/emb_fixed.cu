__global__ void embarrassingly_good(int value) {
    bool even = ( (threadIdx.x%2) == 0 );
    if(even){
        value = A(value);
    }
    for(int i=0; i<4; i++) {
        if((threadIdx.x+i)%2 == 0){
            value = A(value);
        } else {
            value = B(value);
        }
    }
    if(!even){
        value = A(value);
    }
}