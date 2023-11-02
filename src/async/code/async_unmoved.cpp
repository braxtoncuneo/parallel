#include <thread>
#include <chrono>
#include <iostream>
#include <cstdlib>
#include <future>

int dependency(){
    std::this_thread::sleep_for(std::chrono::seconds(1));
    int value = rand();
    std::cout << "The output is "<< value << '\n';
    return value;
}

void dependent(std::future<int> input){
    int value = input.get();
    std::cout << "The input is "<< value << '\n';
}

int main() {
    srand(time(NULL));
    std::future<int>  value_fut = std::async(dependency);
    std::future<void> print_fut = std::async(dependent,value_fut);
    print_fut.get();
    return 0;
}

