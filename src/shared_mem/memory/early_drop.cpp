#include <iostream>
#include <thread>

void process(int *output, int *input) {
    *output = *input;
}

std::thread launch(int *output, int input) {
    return std::thread (process,output,&input);
}

int main () {
    int output = 0;
    int input  = 100;
    std::thread processor = launch (&output,input);
    std::cout << "Launched thread" << std::endl;
    processor.join();
    std::cout << "Output is " << output << std::endl;
    return 0;
}

