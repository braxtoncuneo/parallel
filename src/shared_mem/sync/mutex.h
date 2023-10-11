class mutex {

    bool volatile wants_turn[2];
    int  volatile turn;

    public:

    mutex() {
        turn = 0;
        wants_turn[0] = true;
        wants_turn[1] = true;
    }

    void lock(int thread_id) {
        int other_id = 1 - thread_id;
        wants_turn[thread_id] = true;
        turn = other_id;
        asm volatile("mfence");
        while ((turn != thread_id) && (wants_turn[other_id])){}
    }

    void unlock(int thread_id) {
        wants_turn[thread_id] = false;
        turn = 1 - thread_id;
    }
};