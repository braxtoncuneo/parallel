#include "condition.h"
#include <thread>
#include <chrono>
#include <iostream>
#include <cstdlib>

void dependency(int* output,condition_variable* cond){
    std::this_thread::sleep_for(std::chrono::seconds(1));
    *output = rand();
    std::cout << "The output is "<< *output << '\n';
    cond->notify_one();
}
void dependent(int* input, condition_variable* cond){
    std::mutex mut;
    std::unique_lock<std::mutex> ulock(mut);
    cond->wait(ulock);
    std::cout << "The input is "<< *input << '\n';
}
int main() {
    srand(time(NULL));
    int value;
    condition_variable cond;
    std::thread thread_a(dependency,&value,&cond);
    std::thread thread_b(dependent, &value,&cond);
    thread_a.join();
    thread_b.join();
    return 0;
}