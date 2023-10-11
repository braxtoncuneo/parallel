#include <mutex>
#include <thread>
#include <iostream>
#include "semaphore.h"

void writer(int* array, int size, semaphore*space, semaphore*filled) {
    int index = 0;
    for(int i=0; i<10; i++){
        space->acquire();
        int written =  rand() % 10;
        printf("(%d)",written);
        array[index] = written;
        index = (index+1)%size;
        filled->release();
    }
}

void reader(int* array, int size, semaphore*space, semaphore*filled) {
    int index = 0;
    for(int i=0; i<10; i++){
        filled->acquire();
        printf("[%d]",array[index]);
        index = (index+1)%size;
        space->release();
    }
    std::cout << '\n';
}



int main() {

    int* array = new int[5];
    semaphore space(5);
    semaphore filled(0);

    std::thread read(reader,array,5,&space,&filled);
    std::thread write(writer,array,5,&space,&filled);

    read.join();
    write.join();

    return 0;
}