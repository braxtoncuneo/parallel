#include <mpi.h>
#include <iostream>
#include <string>
#include <cstdlib>
#include <ctime>

// Helper function that prints the input array's values
// on one line, with commas seperating each element
void print_array(int *array, size_t size) {
    for(size_t i=0; i<size; i++) {
        if(i != 0){
            std::cout << ',';
        }
        std::cout << array[i];
    }
    std::cout << '\n';
}


int main(int argc, char* argv[]) {

    // The usual program setup
    MPI_Init(&argc,&argv);
    int process_count;
    MPI_Comm_size(MPI_COMM_WORLD, &process_count);
    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    // Seed RNG
    srand(time(0)*my_rank);

    // Generate array with random size.
    // All elements are equal to the rank, to make the origin of the
    // elements obvious in the output shown.
    size_t size = rand() % 4;
    int* input = new int[size];
    for(size_t i=0; i<size; i++){
        input[i] = my_rank;
    }

    // Use prefix sum to assign offsets
    int offset;
    MPI_Scan(&size,&offset,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);

    if (my_rank == 0) {
        // Use MPI_Gather to construct count/displ arrays for MPI_Gatherv
        int counts[process_count];
        MPI_Gather(&size  ,1,MPI_INT,counts  ,1,MPI_INT,0,MPI_COMM_WORLD);

        // The scan should be offset by one for the displs, since we want
        // the displacement to be the sum of all previous sizes, excluding
        // the size of the array being displaced
        int displs[process_count+1];
        displs[0] = 0;
        MPI_Gather(&offset,1,MPI_INT,displs+1,1,MPI_INT,0,MPI_COMM_WORLD);

        // Allocate array to contain combined sequence, the final
        // displ element (not corresponding to any sub-array) is the size
        size_t full_size = displs[process_count];
        int *combined = new int[full_size];
        // Combine arrays
        MPI_Gatherv(input,size,MPI_INT,combined,counts,displs,MPI_INT,0,MPI_COMM_WORLD);
        // Print then free combined array
        print_array(combined,full_size);
        delete[] combined;
    } else {
        // Use MPI_Gather to construct count/displ arrays for MPI_Gatherv
        MPI_Gather(&size  ,1,MPI_INT,nullptr,1,MPI_INT,0,MPI_COMM_WORLD);
        MPI_Gather(&offset,1,MPI_INT,nullptr,1,MPI_INT,0,MPI_COMM_WORLD);
        MPI_Gatherv(input,size,MPI_INT,nullptr,nullptr,nullptr,MPI_INT,0,MPI_COMM_WORLD);
    }
    delete[] input;

    MPI_Finalize();
    return 0;

}
