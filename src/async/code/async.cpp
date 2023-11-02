#include <thread>
#include <chrono>
#include <iostream>
#include <cstdlib>
#include <future>
#include <functional>

int dependency(){
    std::this_thread::sleep_for(std::chrono::seconds(1));
    int value = rand();
    std::cout << "The output is "<< value << '\n';
    return value;
}

void dependent(std::future<int>* input){
    int value = input->get();
    std::cout << "The input is "<< value << '\n';
}

int main() {
    srand(time(NULL));
    std::future<int>  my_future = std::async(dependency);
    std::thread thread_b(dependent, &my_future);
    thread_b.join();
    return 0;
}

