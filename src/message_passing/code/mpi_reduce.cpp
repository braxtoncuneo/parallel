#include <mpi.h>
#include <iostream>
#include <cstdlib>
#include <ctime>

int main(int argc, char* argv[]) {

    // The usual program setup
    MPI_Init(&argc,&argv);
    int process_count;
    MPI_Comm_size(MPI_COMM_WORLD, &process_count);
    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);


    // Generate random int arrays across all ranks
    srand(time(nullptr)*my_rank);
    size_t const size = 10;
    int values[size];
    for(size_t i=0; i<size; i++){
        values[i] = rand() % 101;
    }


    if(my_rank == 0){
        int sums[size];
        // Reduce arrays element-wise
        MPI_Reduce(&values,&sums,  size,MPI_INT,MPI_SUM,0,MPI_COMM_WORLD);
        // Calculate and print averages
        for(size_t i=0; i<size; i++){
            if(i!=0){
                std::cout << ',';
            }
            std::cout << sums[i]/((float)process_count);
        }
        std::cout << '\n';
    } else {
        MPI_Reduce(&values,nullptr,size,MPI_INT,MPI_SUM,0,MPI_COMM_WORLD);
    }


    MPI_Finalize();
    return 0;

}
