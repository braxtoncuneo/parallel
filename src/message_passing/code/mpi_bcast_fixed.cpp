#include <mpi.h>
#include <iostream>
#include <string>

struct MyStruct {
    int a;
    float b;
    double c[10245];
};


int main(int argc, char* argv[]) {

    MPI_Init(&argc,&argv);

    int process_count;
    MPI_Comm_size(MPI_COMM_WORLD, &process_count);

    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    if(my_rank == 0){
        std::string input_string;
        std::getline(std::cin,input_string);
        printf("Input for rank %d was '%s'\n",my_rank,input_string.c_str());

        size_t message_size = input_string.size() + 1;
        MPI_Bcast(&message_size,sizeof(message_size),MPI_CHAR,0,MPI_COMM_WORLD);
        MPI_Bcast((void*)input_string.c_str(),message_size,MPI_CHAR,0,MPI_COMM_WORLD);
    } else {
        size_t message_size;
        MPI_Bcast(&message_size,sizeof(message_size),MPI_CHAR,0,MPI_COMM_WORLD);
        char *message_buffer = new char[message_size];
        MPI_Bcast((void*)message_buffer,message_size,MPI_CHAR,0,MPI_COMM_WORLD);
        printf("Rank %d recieved input '%s'\n",my_rank,message_buffer);
        delete[] message_buffer;
    }

    MPI_Finalize();
    return 0;
}
