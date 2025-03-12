#ifndef WORKER_THREAD_HPP
#define WORKER_THREAD_HPP

#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <atomic>
namespace MNN {

struct ThreadMsg;

// A singleton instance background worker thread.
class WorkerThread {
public:
    // Post a task to the thread queue.
    // @param task - thread specific message information
    // @return true if the `task` is successfully added to the work queue.
    // PostTask fails if there are more than `MAX_TASKS` tasks in the queue.
    bool postTask(std::function<int()>&& task);

    WorkerThread(int numberThread = 1);
    ~WorkerThread();
private:
    struct Task {
        std::function<void()> content;
    };
    std::vector<std::thread> mWorkers;
    std::atomic<bool> mStop = {false};

    std::queue<Task*> mTasks;
    std::condition_variable mCondition;
    std::mutex mQueueMutex;
    std::mutex mConditionMutex;
};

} // namespace MNN

#endif  // WORKER_THREAD_HPP
