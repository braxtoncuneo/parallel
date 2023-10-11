#include <iostream>
#include <thread>
#include "latch.h"

using std::chrono::milliseconds;

void latched_printer(size_t thread_id,latch *print_latch) {
    uint8_t ms_sleep_count = rand() % 1000;
    std::this_thread::sleep_for(milliseconds(ms_sleep_count));
    std::cout << (char)('A'+thread_id);
    print_latch->arrive_and_wait();
    if(thread_id == 0){
        std::cout << '\n';
    }
}


int main() {
    size_t thread_count = 10;
    std::cout << "At least "<<(thread_count/2)
              <<" characters should appear on the next line\n";
    srand(time(0));
    std::thread team[thread_count];
    latch print_latch(thread_count/2);
    for(size_t i=0; i<thread_count; i++){
        team[i] = std::thread(latched_printer,i,&print_latch);
    }
    for(size_t i=0; i<thread_count; i++){
        team[i].join();
    }
    std::cout << "\n";
}
