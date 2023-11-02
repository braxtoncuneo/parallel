#include <thread>
#include <chrono>
#include <iostream>
#include <cstdlib>
#include <future>

void dependency(std::promise<int>* output){
    std::this_thread::sleep_for(std::chrono::seconds(1));
    int value = rand();
    std::cout << "The output is "<< value << '\n';
    output->set_value(value);
}

void dependent(std::future<int>* input){
    int value = input->get();
    std::cout << "The input is "<< value << '\n';
}

int main() {
    srand(time(NULL));
    std::promise<int> my_promise;
    std::future<int>  my_future = my_promise.get_future();
    std::thread thread_a(dependency,&my_promise);
    std::thread thread_b(dependent, &my_future);
    thread_a.join();
    thread_b.join();
    return 0;
}