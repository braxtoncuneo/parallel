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

void dependent_a(std::future<int>* input){
    int value = input->get();
    std::cout << "The input plus 1 is "<< (value+1) << '\n';
}

void dependent_b(std::future<int>* input){
    int value = input->get();
    std::cout << "The input plus 2 is "<< (value+2) << '\n';
}

int main() {
    srand(time(NULL));
    std::future<int>  value_fut = std::async(dependency);
    std::future<void> a_fut     = std::async(dependent_a,&value_fut);
    std::future<void> b_fut     = std::async(dependent_b,&value_fut);
    a_fut.get();
    b_fut.get();
    return 0;
}

