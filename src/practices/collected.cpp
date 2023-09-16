#include <iostream>
#include <chrono>
#include <thread>
#include <cstdlib>

using std::chrono::steady_clock;
using std::chrono::milliseconds;
using std::chrono::duration_cast;
using std::this_thread::sleep_for;
using time_point = std::chrono::steady_clock::time_point;
using time_span = std::chrono::duration<double>;


int main(int argc, char *argv[]) {

    if(argc != 3){
        std::cerr << "Usage: program [add_count] [mul_count]\n";
        return 1;
    }

    int add_count = atoi(argv[1]);
    int mul_count = atoi(argv[2]);

    if( (add_count < 0) || (mul_count < 0) ){
        std::cerr << "All inputs must be positive integers.\n";
        return 2;
    }


    time_point start_time = steady_clock::now();

    // Sum the products of many random numbers
    int sum = 0;
    for(int i=0; i<add_count; i++) {
        int product = 1;
        for(int j=0; j < mul_count; j++) {
            product *= rand() % 10;
        }
        sum += product;
    }

    time_point end_time = steady_clock::now();

    time_span span = duration_cast<time_span>(end_time-start_time);

    std::cout << span.count();
    
    return 0;
}
