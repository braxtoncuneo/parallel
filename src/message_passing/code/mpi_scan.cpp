#include <mpi.h>
#include <iostream>
#include <string>
#include <cstdlib>

// Helper function that generates an array of random
// integers between 0 and 9
int* random_array(size_t size) {
    int *result = new int[size];
    for(int i=0; i<size; i++){
        result[i] = rand() % 10;
    }
    return result;
}

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

// Helper function that concatenates equally-sized arrays
// across all processes and prints the combined array
void gather_print_array(int* array, size_t size){

    // Re-find the process count and rank, to avoid
    // additional function parameters
    int process_count;
    MPI_Comm_size(MPI_COMM_WORLD, &process_count);
    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    // Allocate array at root
    int *full_array = nullptr;
    if(my_rank == 0){
        full_array = new int[size*process_count];
    }

    // Combine each rank's array
    MPI_Gather(array,size,MPI_INT,full_array,size,MPI_INT,0,MPI_COMM_WORLD);

    // Print combined array, then clean it up
    if(my_rank == 0){
        print_array(full_array,size*process_count);
        delete[] full_array;
    }

}

int main(int argc, char* argv[]) {

    // The usual program setup
    MPI_Init(&argc,&argv);
    int process_count;
    MPI_Comm_size(MPI_COMM_WORLD, &process_count);
    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    // Guard against bad input
    if(argc != 2){
        if(my_rank == 0){
            std::cout << "Error: program requires exactly one input\n";
        }
        MPI_Finalize();
        return 1;
    }

    // Elements per rank
    size_t size = atoi(argv[1]);

    // Generate and print elements across all ranks
    srand(my_rank);
    int* input = random_array(size);
    gather_print_array(input,size);

    // Scan all arrays across all ranks
    int *output = new int[size];
    MPI_Scan(input,output,size,MPI_INT,MPI_SUM,MPI_COMM_WORLD);

    // Print the prefix-scan output
    gather_print_array(output,size);
    delete[] output;

    MPI_Finalize();
    return 0;

}
