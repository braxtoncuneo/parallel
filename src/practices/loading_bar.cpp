#include <iostream>
#include <chrono>
#include <thread>

// So we don't have to resolve the scope every time
using std::chrono::steady_clock;
using std::chrono::milliseconds;
using std::chrono::duration_cast;
using std::this_thread::sleep_for;
using TimePoint = std::chrono::steady_clock::time_point;
using TimeSpan = std::chrono::duration<double>;

int main() {

    std::cout << "Here's a fun loading bar:\n\n";

    // Get current time before we do something
    TimePoint start_time = steady_clock::now();

    // A fun animation while we wait
    for(int i=0; i<40; i++){
        std::cout << '|';
        sleep_for(milliseconds(10));
        std::cout.flush();
    }

    // Get current time after we are done
    TimePoint end_time = steady_clock::now();

    // Convert the difference of the times to get a duration
    TimeSpan span = duration_cast<time_span>(end_time-start_time);

    std::cout << "\n\nIt took "<< span.count() <<" seconds to load.\n";
    return 0;
}