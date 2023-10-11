#include <iostream>
#include <thread>

void adder(int* total, mutex* mut, int thread_id) {
    int limit = 1000000;
    for(int i=0; i<limit; i++) {
        mut->lock(thread_id);
        *total += 1;
        mut->unlock(thread_id);
    }
}

int main() {
    int total = 0;
    mutex mut;
    std::thread thread_a(adder,&total,&mut,0);
    std::thread thread_b(adder,&total,&mut,1);
    thread_a.join();
    thread_b.join();
    std::cout << "Total is "<<total<<'\n';
    return 0;
}