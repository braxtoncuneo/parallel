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

        int value = 1234;
        for(int i=1; i<process_count; i++){
            MPI_Send(&value,1,MPI_INT,i,0,MPI_COMM_WORLD);
        }

    } else {

        MPI_Status status;
        int value;
        MPI_Recv(&value,1,MPI_INT,0,MPI_ANY_TAG,MPI_COMM_WORLD,&status);
        printf("Rank %d recieved int %d from rank %d with tag\n",my_rank,value,status.MPI_SOURCE,status.MPI_TAG);

    }

    MPI_Finalize();
    return 0;
}