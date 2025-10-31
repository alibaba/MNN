#pragma once

#include <deque>
#include <thread>

#ifdef NUM_THREAD_WORKERS
static const size_t kNumThreadWorkers = NUM_THREAD_WORKERS;
#else
static const size_t kNumThreadWorkers = std::thread::hardware_concurrency();
#endif

// A thread pool that only ensures there are at most N number of active threads in the pool,
// without any regard to thread creation/destruction overhead.
// However, if `numWorkers` is set to 0, then there is no limit to the number of active threads.
class BasicThreadPool {
public:
    BasicThreadPool(const size_t numWorkers) : kNumWorkers(numWorkers) {}

    BasicThreadPool() : kNumWorkers(kNumThreadWorkers) {}

    ~BasicThreadPool() {
        // Ensure all threads are joined before destroying itself
        joinAll();
    }

    bool empty() const { return mThreadPool.empty(); }

    template <class Func, class... Args>
    void push(Func&& func, Args&&... args) {
        // Create a new thread and start executing it
        mThreadPool.emplace_back(std::forward<Func>(func), std::forward<Args>(args)...);

        // Join the threads once the number of active threads have reached `kNumWorkers`.
        // However if `kNumWorkers` is 0, allow unlimited number of active threads.
        if (kNumWorkers && mThreadPool.size() == kNumWorkers)
            joinAll();
    }

    void joinAll() {
        for (auto& thread : mThreadPool)
            thread.join();
        mThreadPool.clear();
    }

private:
    const size_t kNumWorkers;
    std::deque<std::thread> mThreadPool;
};