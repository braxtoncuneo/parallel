#include <mpi.h>
#include <iostream>
#include <string>

int main(int argc, char* argv[]) {

    MPI_Init(&argc,&argv);

    int process_count;
    MPI_Comm_size(MPI_COMM_WORLD, &process_count);

    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    // Only rank 0 gets stdin from our terminal,
    // so let's only try to read input from rank 0.
    if(my_rank == 0){
        std::string input_string;
        std::getline(std::cin,input_string);
        printf("Input for rank %d was '%s'\n",my_rank,input_string.c_str());
    }

    MPI_Finalize();
    return 0;
}
