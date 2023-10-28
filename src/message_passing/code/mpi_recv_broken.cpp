#include <mpi.h>
#include <iostream>
#include <string>

int main(int argc, char* argv[]) {

    MPI_Init(&argc,&argv);

    int process_count;
    MPI_Comm_size(MPI_COMM_WORLD, &process_count);

    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    if (my_rank == 0) {
        int value = 12345;
        MPI_Send(&value,1,MPI_INT,my_rank+1,0,MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}
