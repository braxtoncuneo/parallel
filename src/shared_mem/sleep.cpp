#include <iostream>
#include <thread>
#include <chrono>
#include <condition_variable>

using std::chrono::milliseconds;

void dependent(std::condition_variable *cond) {
    std::cout <<  "Hello from the dependent!\n";
    cond->wait();
    std::cout <<  "Goodbye from the dependent!\n";
}

void dependee(std::condition_variable *cond) {
    std::cout <<  "Hello from the dependee!\n";
    std::this_thread::sleep_for(milliseconds(10));
    std::cout <<  "Goodbye from the dependee!\n";
    cond->notify_one();
}

int main() {
    std::condition_variable cond;
    std::thread a(dependent,&cond);
    std::thread b(dependee, &cond);
    a.join();
    b.join();
    return 0;
}