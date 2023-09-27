#include <iostream>
#include <thread>
#include <chrono>

// Waits until the user hits enter, then sets the bool pointed
// by `user_hit_enter` to true.
void watcher (bool* user_hit_enter){
    std::string dummy;
    std::getline(std::cin,dummy);
    *user_hit_enter = true;
}


int main (int argc, char *argv[]){

    // Initialize input flag and spawn watcher thread
    bool user_hit_enter = false;
    std::thread watcher_thread(watcher,&user_hit_enter);
    watcher_thread.detach();

    // Determine the message to display
    std::string message;
    if(argc <= 1){
        message = "Your message here";
    } else {
        message = argv[1];
    }

    // Loop through and print element rotations of the message
    // until the watcher thread notifies us of input.
    int position = 0;
    while(!user_hit_enter){
        std::cout << message.substr(position);
        std::cout << message.substr(0,position);
        std::cout << '\r';
        // Flush to prevent buffering from hurting our frame rate
        std::flush(std::cout);
        position = (position + 1) % message.size();
        // Update the display every tenth of a second
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    std::cout << '\n';

    return 0;
}