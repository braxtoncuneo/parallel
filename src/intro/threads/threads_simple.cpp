#include <iostream>
#include <thread>
#include <chrono>
#include <string>

void print_number(int value) {
    std::cout << value << '\n';
}

void print_string(std::string text ) {
    // adding it beforehand to avoid a race condition
    text += '\n';
    std::cout << text;
}

int main() {

    std::thread joined_thread(print_number,1234);
    std::thread detached_thread(print_string,"Hello!");
    
    detached_thread.detach();
    
    std::this_thread::sleep_for(std::chrono::seconds(1));
    print_string("Greetings!");
    std::this_thread::sleep_for(std::chrono::seconds(1));

    joined_thread.join();

    return 0;
}