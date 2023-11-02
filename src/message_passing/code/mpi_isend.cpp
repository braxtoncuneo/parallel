#include <mpi.h>
#include <iostream>
#include <string>

int main(int argc, char* argv[]) {

    MPI_Init(&argc,&argv);

    int process_count;
    MPI_Comm_size(MPI_COMM_WORLD, &process_count);

    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    MPI_Request send_req;
    MPI_Status  status;

    int value = (my_rank == 0) ? 12345 : 67890;

    if( (my_rank >=0) && (my_rank <= 1) ) {
        int other_rank = 1 - my_rank;
        MPI_Isend(&value,1,MPI_INT,other_rank,0,          MPI_COMM_WORLD,&send_req);
        MPI_Recv (&value,1,MPI_INT,other_rank,MPI_ANY_TAG,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        printf("Rank %d recieved int %d\n",my_rank,value);
    }

    MPI_Finalize();
    return 0;
}