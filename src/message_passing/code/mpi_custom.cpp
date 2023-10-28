#include <mpi.h>
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cmath>

struct Vec2D {
    float x;
    float y;
};

MPI_Datatype Vec2DType;

float magnetude(Vec2D point){
    return sqrt(point.x*point.x + point.y*point.y);
}

// Finds the 2D vector with the greatest magnetude
void max_magnetude(void *invec, void *inoutvec, int *len, MPI_Datatype *datatype) {
    // To keep things simple, this function only works with Vec2D values, so
    // the function should abort with a type error if any other type is supplied.
    if(*datatype != Vec2DType) {
        MPI_Abort(MPI_COMM_WORLD,MPI_ERR_TYPE);
    }
    // Cast/dereference the input pointers to something more reasonable
    Vec2D *in    = (Vec2D*) invec;
    Vec2D *inout = (Vec2D*) inoutvec;
    int length = *len;
    for(int i=0; i<length; i++){
        float left_mag  = magnetude(in[i]);
        float right_mag = magnetude(inout[i]);
        if(left_mag > right_mag){
            inout[i] = in[i];
        }
    }
}


int main(int argc, char* argv[]) {

    // The usual program setup
    MPI_Init(&argc,&argv);
    int process_count;
    MPI_Comm_size(MPI_COMM_WORLD, &process_count);
    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    // Declare a type handle
    int blocklengths[2]   = { 1, 1 };
    MPI_Aint displacements[2]  = { offsetof(Vec2D,x), offsetof(Vec2D,y) };
    MPI_Datatype types[2] = { MPI_FLOAT, MPI_FLOAT };
    MPI_Type_create_struct(2,blocklengths,displacements,types,&Vec2DType);

    MPI_Type_commit(&Vec2DType);

    // Declare an op handle
    MPI_Op max_mag;
    // Initialize the op handle using the custom sum function
    MPI_Op_create(max_magnetude,1,&max_mag);


    // Generate random 2D vectors across all ranks
    srand(time(nullptr)*my_rank);
    Vec2D my_vector;
    my_vector.x = (rand()%1000) / 1000.0;
    my_vector.y = (rand()%1000) / 1000.0;
    float my_mag = magnetude(my_vector);
    printf("Rank %d vector: (%g,%g) with magnetude %g\n",my_rank, my_vector.x,my_vector.y,my_mag);

    if(my_rank == 0){
        Vec2D biggest_vector;
        // Reduce arrays element-wise
        MPI_Reduce(&my_vector,&biggest_vector,1,Vec2DType,max_mag,0,MPI_COMM_WORLD);
        float mag = magnetude(biggest_vector);
        printf("Biggest vector: (%g,%g) with magnetude %g\n",biggest_vector.x,biggest_vector.y,mag);
    } else {
        MPI_Reduce(&my_vector,nullptr,1,Vec2DType,max_mag,0,MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;

}
