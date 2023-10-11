#include <mutex>
#include <vector>
class condition_variable {
    std::mutex lock;
    std::vector<bool*> waiting_threads; 
    public:

    void wait (std::unique_lock<std::mutex>& waiter_lock) {
        bool flag = false;
        lock.lock();
        waiter_lock.unlock();
        waiting_threads.push_back(&flag);
        lock.unlock();
        while(!flag){}
        waiter_lock.lock();
    }

    void notify_one () {
        lock.lock();
        bool* waiter_flag = waiting_threads.back();
        waiting_threads.pop_back();
        *waiter_flag = true;
        lock.unlock();
    }

    void notify_all () {
        lock.lock();
        while(!waiting_threads.empty()){
            bool* waiter_flag = waiting_threads.back();
            waiting_threads.pop_back();
            *waiter_flag = true;
        }
        lock.unlock();
    }
};