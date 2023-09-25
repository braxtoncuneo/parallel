#include <iostream>
#include <thread>
#include <chrono>

void print (bool* user_hit_enter){
    std::string dummy;
    std::getline(std::cin,dummy);
    *user_hit_enter = true;
}


int main (int argc, char *argv[]){

    bool user_hit_enter = false;
    std::thread print_thread(print,&user_hit_enter);
    std::string message;

    if(argc <= 1){
        message = "Your message here";
    } else {
        message = argv[1];
    }

    int position = 0;
    while(!user_hit_enter){
        std::cout << message.substr(position);
        std::cout << message.substr(0,position);
        std::cout << '\r';
        std::flush(std::cout);
        position = (position + 1) % message.size();
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    std::cout << '\n';

    print_thread.join();

    return 0;
}