class mutex {
    int  turn;
    public:
    mutex() {
        turn = 0;
    }
    void lock(int thread_id) {
        int other_id = 1 - thread_id;
        while (turn != thread_id){}

    }
    void unlock(int thread_id) {
        turn = 1 - thread_id;
    }
};