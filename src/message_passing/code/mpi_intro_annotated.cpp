#include <iostream>
#include <mpi.h>

int main(int argc, char* argv[]) {

    // Sets up the MPI context. Processors coordinate with
    // each-other to figure out how they communicate, who
    // is who, etc. Can only be called once.
    MPI_Init(&argc,&argv);
    //          ^   ^
    // On some MPI implementations, supplying argc/argv will
    // cause MPI to remove MPI-related flags, making the
    // parsing of non-MPI args easier. If you aren't on one
    // of those implementations, or you don't care, you can
    // just give null pointers, like this:
    // MPI_Init(nullptr,nullptr);

    // "Returns the size of the group associated with the communicator"
    // - https://www.open-mpi.org/doc/v3.1/man3/MPI_Comm_size.3.php
    // In other words, it gives you the number of processes
    int process_count;
    MPI_Comm_size(MPI_COMM_WORLD, &process_count);

    // "Determines the rank of the calling process in the communicator."
    // - https://www.open-mpi.org/doc/v4.0/man3/MPI_Comm_rank.3.php
    // In other words, it gets an integer that uniquely identifies
    // the process that made the call.
    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    // For demonstration, let's have each process print their rank
    printf("My rank (aka ID) is %d out of %d total ranks\n",my_rank,process_count);

    // Cleans up the MPI context. Can only be called once.
    // After being called, no more communication can be
    // performed.
    MPI_Finalize();
    return 0;
}
