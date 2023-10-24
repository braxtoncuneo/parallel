#include <mpi.h>
#include <iostream>
#include <string>

int main(int argc, char* argv[]) {

    MPI_Init(&argc,&argv);

    int process_count;
    MPI_Comm_size(MPI_COMM_WORLD, &process_count);

    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    if( (my_rank%2 == 0) &&  (my_rank < process_count-1) ){
        char my_char = 'a' + my_rank;
        MPI_Send(&my_char,1,MPI_CHAR,0,MPI_COMM_WORLD);
    } else {

    }

    MPI_Finalize();
    return 0;
}
