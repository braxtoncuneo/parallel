#include <iostream>
#include <string>
#include <functional>


class Scheduler {

    std::set<UserThread>     blocked;
    std::dequeue<UserThread> run_queue;

    public:

    

};

template <typename RETURN_TYPE, typename... ARG_TYPES>
struct UserThread {

    std::function<RETURN_TYPE()> callback;

    UserThread( RETURN_TYPE(&&function)(ARG_TYPES...), ARG_TYPES&&... args)
        : callback(std::bind(function, args...))
    {}

    decltype(auto) call () {
        return callback();
    }

};



int function_a(int x) {
    std::cout << "Hello from function A!\n";
    return 1234;
}

std::string function_b(int x) {
    std::cout << "Hello from function B!\n";
    return "Result string";
}


int main(){
    UserThread thread_a(function_a,0);
    UserThread thread_b(function_b,0);

    std::cout << "Executing Thread B...\n";
    std::string b_result = thread_b.call();
    std::cout << "Thread B returned: " << b_result << '\n';

    std::cout << "Executing thread A...\n";
    int a_result = thread_a.call();
    std::cout << "Thread A returned: " << a_result << '\n';

    return 0;
}
