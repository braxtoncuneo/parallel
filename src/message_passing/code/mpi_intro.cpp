#include <iostream>
#include <mpi.h>

int main(int argc, char* argv[]) {

    MPI_Init(&argc,&argv);

    int process_count;
    MPI_Comm_size(MPI_COMM_WORLD, &process_count);

    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    printf("My rank (aka ID) is %d out of %d total ranks\n",my_rank,process_count);

    MPI_Finalize();
    return 0;
}
