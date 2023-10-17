#include <mutex>
#include "condition.h"
class semaphore {
    int total;
    int counter;
    std::mutex lock;
    condition_variable cond;
    public:

    semaphore(int total)
        : total(total)
        , counter(total)
    {}

    void acquire() {
        std::unique_lock<std::mutex> ulock(lock);
        counter--;
        if(counter < 0){
            cond.wait(ulock);
        }
        lock.unlock();
    }

    void release() {
        lock.lock();
        counter++;
        if(counter <=0){
            cond.notify_one();
        }
        lock.unlock();
    }

    void release_all() {
        lock.lock();
        counter = total;
        cond.notify_all();
        lock.unlock();
    }
};