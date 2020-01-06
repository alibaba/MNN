//
//  BlockingQueue.hpp
//  MNN
//
//  Created by MNN on 2019/11/19.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef BlockingQueue_hpp
#define BlockingQueue_hpp
#include <MNN/MNNDefine.h>
#include <condition_variable>
#include <mutex>
#include <queue>

namespace MNN {
namespace Train {

template <typename T>
class BlockingQueue {
public:
    BlockingQueue() = default;
    BlockingQueue(size_t maxSize) : mMaxSize(maxSize) {
    }

    bool isFull() {
        return mQueue.size() == mMaxSize;
    }

    bool isEmpty() {
        return mQueue.empty();
    }

    void push(T value) {
        {
            std::unique_lock<std::mutex> lock(mMutex);
            mCondVar.wait(lock, [&] { return !isFull(); });
            MNN_ASSERT(!isFull());
            mQueue.push(std::move(value));
            lock.unlock();
        }
        mCondVar.notify_one();
    }

    T pop() {
        std::unique_lock<std::mutex> lock(mMutex);
        mCondVar.wait(lock, [&] { return !isEmpty(); });
        MNN_ASSERT(!isEmpty());
        T value = mQueue.front();
        mQueue.pop();
        mCondVar.notify_one();
        lock.unlock();

        return std::move(value);
    }

    size_t clear() {
        std::lock_guard<std::mutex> lock(mMutex);
        const auto size = mQueue.size();
        while (!isEmpty()) {
            mQueue.pop();
        }
        return size;
    }

private:
    size_t mMaxSize;
    std::queue<T> mQueue;
    std::mutex mMutex;
    std::condition_variable_any mCondVar;
};

} // namespace Train
} // namespace MNN

#endif // BlockingQueue_hpp
