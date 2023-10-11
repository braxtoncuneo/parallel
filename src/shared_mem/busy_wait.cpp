#include <iostream>
#include <thread>
#include <chrono>

using std::chrono::milliseconds;

void dependent(bool *flag) {
    std::cout <<  "Hello from the dependent!\n";
    while(!*flag){}
    std::cout <<  "Goodbye from the dependent!\n";
}

void dependee(bool *flag) {
    std::cout <<  "Hello from the dependee!\n";
    std::this_thread::sleep_for(milliseconds(10));
    std::cout <<  "Goodbye from the dependee!\n";
    *flag = true;
}

int main() {
    bool flag = false;
    std::thread a(dependent,&flag);
    std::thread b(dependee, &flag);
    a.join();
    b.join();
    return 0;
}