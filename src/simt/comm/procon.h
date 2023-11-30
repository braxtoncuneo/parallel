template<typename T>
class ProCon {

    T* array;
    int  size;
    int *iter;

    public:

    // Construction is funky on CUDA. To make construction
    // more explicit, it is represented by a static method.
    static ProCon<T> create(int size) {
        ProCon<T> result;
        auto_throw(cudaMallocManaged(&result.array,sizeof(T)*size));
        auto_throw(cudaMallocManaged(&result.iter, sizeof(int)));
        result.size = size;
        *result.iter = 0;
        return result;
    }

    __device__ void give(T data) {
        int index = atomicAdd(iter,1);
        if(index >= size){
            // Guard for the error message, so the message prints
            // only once after the capacity is exceeded
            if(index == size){
                printf("ERROR: ProCon capacity exceeded!\n");
            }
            return;
        }
        array[index] = data;
    }

    __device__ bool take(T& data){
        int index = atomicAdd(iter,-1);
        if(index < 1){
            return false;
        }
        data = array[index-1];
        return true;
    }

    // Destruction is also weird on CUDA
    void destruct() {
        auto_throw(cudaFree(array));
        auto_throw(cudaFree(iter));
    }

};