#include <iostream>
#include <chrono>


// So we don't have to resolve the scope every time
using std::chrono::steady_clock;
using std::chrono::duration_cast;
using std::chrono::steady_clock::time_point;
using time_span = std::chrono::duration<double>;

int main() {

    std::cout << "Here's a fun loading bar:\n\n\n";

    // Get current time before we do something
    time_point start_time = steady_clock::now();

    // A fun animation while we wait
    for(int i=0; i<=100; i++){
        std::cout << '[';
        for(int j=0; j<100; j+=1){
            std::cout << (j<i) ? '|' : ' ';
        }
        std::cout << "]\r";
    }

    // Get current time after we are done
    time_point end_time = steady_clock::now();

    // Convert the difference of the times to get a duration
    time_span span = duration_cast<time_span>(start_time-end_time);

    std::cout << "\n\nIt took "<< span.count() <<" seconds to load.\n";
    return 0;
}