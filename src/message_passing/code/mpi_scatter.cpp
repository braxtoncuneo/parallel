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
        MPI_Finalize();
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

    // Generate count and displ arrays for the MPI_Scatterv
    // and MPI_Gatherv calls
    int counts[process_count];
    int displs[process_count];
    for(int i=0; i<process_count; i++){
        size_t start = chunk_size*i;
        start = (start > length) ? length : start;
        size_t end   = chunk_size*(i+1);
        end   = (end > length) ? length : end;
        size_t size = end-start;
        counts[i] = end-start;
        displs[i] = start;
    }

    // Scatter the text
    size_t my_size = counts[my_rank];
    char  *my_text = new char[my_size];
    MPI_Scatterv(text,counts,displs,MPI_CHAR,my_text,my_size,MPI_CHAR,0,MPI_COMM_WORLD);

    // Capitalize our portion of the text
    for(size_t i=0; i<my_size; i++){
        if( (my_text[i] >= 'a') && (my_text[i] <= 'z') ){
            my_text[i] -= 'a'-'A';
        }
    }

    // Gather the text, then clean up the
    // local text array
    MPI_Gatherv(my_text,my_size,MPI_CHAR,text,counts,displs,MPI_CHAR,0,MPI_COMM_WORLD);
    delete[] my_text;

    // Print the output
    if(my_rank == 0){
        std::cout << text << '\n';
    }
    delete[] text;

    MPI_Finalize();
    return 0;
}
