#include <iostream>
#include <thread>
#include <vector>

#define CONDITION 1
#define SEMAPHORE 1
#define LATCH     1


#if CONDITION == 1
#include "condition.h"
#else
#include <condition_variable>
using std::condition_variable;
#endif


#if SEMAPHORE == 1
#include "semaphore.h"
#else
#include <semaphore>
using std::semaphore;
#endif



#if LATCH == 1
#include "latch.h"
#else
#include <latch>
using std::latch;
#endif






void writer(int* array, int size, semaphore*space, semaphore*filled) {
    int index = 0;
    for(int i=0; i<10; i++){
        space->wait();
        int written =  rand() % 10;
        std::cout << "(" << written << ")";
        array[index] = written;
        index = (index+1)%size;
        filled->signal();
    }
}

void reader(int* array, int size, semaphore*space, semaphore*filled) {
    int index = 0;
    for(int i=0; i<10; i++){
        filled->wait();
        std::cout << array[index] << ", ";
        index = (index+1)%size;
        space->signal();
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
    /*
    size_t thread_count = 2;
    int turn  = 0;
    int total = 0;
    int wants_turn[2] = {true, true};
    std::thread team[thread_count];
    for(int i=0; i<thread_count; i++){
        team[i] = std::thread(adder,&total,&sem);
    }
    for(int i=0; i<thread_count; i++){
        team[i].join();
    }
    std::cout << "The total is "<< total << '\n';

    */
    return 0;
}



