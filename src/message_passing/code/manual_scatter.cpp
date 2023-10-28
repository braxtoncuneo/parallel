#include <mpi.h>
#include <iostream>
#include <string>
#include <cstring>

int main(int argc, char* argv[]) {

    // The usual program setup
    MPI_Init(&argc,&argv);
    int process_count;
    MPI_Comm_size(MPI_COMM_WORLD, &process_count);
    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    // Guard against bad input
    if(argc != 2){
        std::cout << "Error: Program requires exactly one parameter.\n";
        return 1;
    }

    // Copy the input into a writable array
    size_t length;
    char *text = nullptr;
    if(my_rank == 0){
        length = strlen(argv[1]);
        text = new char[length+1];
        strcpy(text,argv[1]);
    }

    // Let all processes know the size of the input
    MPI_Bcast(&length,sizeof(size_t),MPI_CHAR,0,MPI_COMM_WORLD);

    // Calculate the default amount of text to give to
    // each rank (rounding up for better load balancing)
    size_t chunk_size;
    chunk_size = (length+process_count-1)/process_count;

    // Set up pointer/size for local text array
    char*  my_text = nullptr;
    size_t my_text_size = 0;

    if (my_rank == 0) {
        // Distribute chunks of the text across all ranks
        for(int i=1; i<process_count; i++){
            size_t start = chunk_size*i;
            start = (start > length) ? length : start;
            size_t end   = chunk_size*(i+1);
            end   = (end > length) ? length : end;
            size_t size = end-start;
            MPI_Send(&size,sizeof(size_t),MPI_CHAR,i,0,MPI_COMM_WORLD);
            MPI_Send(&text[start],size, MPI_CHAR,i,0,MPI_COMM_WORLD);
        }
        my_text      = text;
        my_text_size = chunk_size;
    } else {
        // Recieve text from the rank 0 process
        MPI_Recv(&my_text_size,sizeof(size_t),MPI_CHAR,0,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        std::cout.flush();
        my_text = new char[my_text_size];
        MPI_Recv(my_text,my_text_size,MPI_CHAR,0,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
    }

    // Capitalize our portion of the text
    for(size_t i=0; i<my_text_size; i++){
        if( (my_text[i] >= 'a') && (my_text[i] <= 'z') ){
            my_text[i] -= 'a'-'A';
        }
    }


    if (my_rank == 0) {
        // Retrieve the capitalized text chunks from each rank, assembling
        // the final output
        for(int i=1; i<process_count; i++){
            size_t start = chunk_size*i;
            start = (start > length) ? length : start;
            size_t end   = chunk_size*(i+1);
            end   = (end > length) ? length : end;
            size_t size = end-start;
            MPI_Recv(&text[start],size,MPI_CHAR,i,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        }
    } else {
        // Send back capitalized text to the root
        MPI_Send(my_text,my_text_size,MPI_CHAR,0,0,MPI_COMM_WORLD);
        delete[] my_text;
    }

    // Print the output
    if(my_rank == 0){
        std::cout << text << '\n';
    }

    MPI_Finalize();
    return 0;
}
