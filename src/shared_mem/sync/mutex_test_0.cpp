#include <iostream>
#include <thread>

void adder(int* total, int thread_id) {
    int limit = 1000000;
    for(int i=0; i<limit; i++) {
        *total += 1;
    }
}

int main() {
    int total = 0;
    std::thread thread_a(adder,&total,0);
    std::thread thread_b(adder,&total,1);
    thread_a.join();
    thread_b.join();
    std::cout << "Total is "<<total<<'\n';
    return 0;
}