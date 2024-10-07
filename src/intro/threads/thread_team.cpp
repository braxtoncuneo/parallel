#include <iostream>
#include <thread>
#include <chrono>
#include <vector>
#include <string>

void print_message(std::string message, volatile bool &flag) {
    while(flag) {
        std::this_thread::sleep_for(
            std::chrono::seconds(1)
        );
        std::cout << message << std::endl;
    }
}


int main() {

    size_t TEAM_SIZE = 10;

    bool flag = true;
    std::vector<std::thread> team;

    for(size_t i=0; i<TEAM_SIZE; i++) {
        std::string message = std::to_string(i);
        team.emplace_back(std::thread(
            print_message,
            message,
            std::ref(flag)
        ));
    }

    std::string dummy;
    std::getline(std::cin,dummy);
    flag = false;

    for(size_t i=0; i<TEAM_SIZE; i++) {
        team[i].join();
    }

    return 0;
}
