#include <mutex>
#include "semaphore.h"
class latch {
    int counter;
    std::mutex lock;
    semaphore  sem;
    public:

    latch(int total)
        : sem(0)
    {
        counter = total;
    }

    void arrive_and_wait() {
        lock.lock();
        int position = counter--;
        lock.unlock();
        if (position > 0) {
            sem.wait();
        } else if (position == 0) {
            sem.notify_all();
        }
    }
};