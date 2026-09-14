//
//  AutoStorage.h
//  MNN
//
//  Created by MNN on 2018/07/14.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef AutoStorage_h
#define AutoStorage_h

#include <stdint.h>
#include <string.h>
#include <atomic>
#include <type_traits>
#include "MNNMemoryUtils.h"

namespace MNN {
template <typename T>

/** self-managed memory storage */
class AutoStorage {
public:
    /**
     * @brief default initializer.
     */
    AutoStorage() {
        mSize = 0;
        mData = NULL;
    }
    /**
     * @brief parameter initializer.
     * @param size  number of elements.
     */
    AutoStorage(int size) {
        mData = (T*)MNNMemoryAllocAlign(sizeof(T) * size, MNN_MEMORY_ALIGN_DEFAULT);
        mSize = size;
    }
    /**
     * @brief deinitializer.
     */
    ~AutoStorage() {
        if ((NULL != mData) && mRelease) {
            MNNMemoryFreeAlign(mData);
        }
    }

    /**
     * @brief get number of elements.
     * @return number of elements..
     */
    inline int size() const {
        return mSize;
    }

    /**
     * @brief set data with number of elements.
     * @param data  data pointer create with `MNNMemoryAllocAlign`.
     * @param size  number of elements.
     * @warning do NOT call `free` or `MNNMemoryFreeAlign` for data pointer passes in.
     */
    void set(T* data, int size) {
        if (NULL != mData && mData != data) {
            MNNMemoryFreeAlign(mData);
        }
        mData = data;
        mSize = size;
    }

    /**
     * @brief set data with outter pointer, can decide whether free pointer when destructor
     * @param data  data pointer, malloc by outside
     * @param release  whether free pointer when destructor
     * @warning User should ensure memory length is enough
     */
    void set(T* data, bool release) {
        mData = data;
        mRelease = release;
    }
    /**
     * @brief reset data size.
     * @param size  number of elements.
     * @warning writed data won't be kept.
     */
    void reset(int size) {
        if (NULL != mData) {
            MNNMemoryFreeAlign(mData);
        }
        mData = (T*)MNNMemoryAllocAlign(sizeof(T) * size, MNN_MEMORY_ALIGN_DEFAULT);
        mSize = size;
    }

    /**
     * @brief release allocated data.
     */
    void release() {
        if (mRelease && NULL != mData) {
            MNNMemoryFreeAlign(mData);
            mData = NULL;
            mSize = 0;
        }
    }

    /**
     * @brief set allocated memory data to 0.
     */
    void clear() {
        ::memset(mData, 0, mSize * sizeof(T));
    }

    /**
     * @brief get data pointer.
     * @return data pointer.
     */
    T* get() const {
        return mData;
    }

private:
    T* mData  = NULL;
    int mSize = 0;
    bool mRelease = true;
};

/** Auto Release Class*/
template <typename T>
class AutoRelease {
public:
    AutoRelease(T* d = nullptr) {
        mData = d;
    }
    ~AutoRelease() {
        if (NULL != mData) {
            delete mData;
        }
    }
    AutoRelease(const AutoRelease&)  = delete;
    T* operator->() {
        return mData;
    }
    void reset(T* d) {
        if (nullptr != mData) {
            delete mData;
        }
        mData = d;
    }
    T* get() {
        return mData;
    }
    const T* get() const {
        return mData;
    }
private:
    T* mData  = NULL;
};


class RefCount
{
    public:
        void addRef() const
        {
            mNum.fetch_add(1, std::memory_order_relaxed);
        }
        void decRef() const
        {
            int prev = mNum.fetch_sub(1, std::memory_order_acq_rel);
            MNN_ASSERT(prev >= 1);
            if (1 == prev)
            {
                delete this;
            }
        }
    inline int count() const{return mNum.load(std::memory_order_relaxed);}
    protected:
        RefCount():mNum(1){}
        RefCount(const RefCount& f):mNum(f.mNum.load(std::memory_order_relaxed)){}
        void operator=(const RefCount& f)
        {
            if (this != &f)
            {
                mNum.store(f.mNum.load(std::memory_order_relaxed), std::memory_order_relaxed);
            }
        }
        virtual ~RefCount(){}
    private:
        mutable std::atomic<int> mNum;
        // Must stay atomic: SharedPtr-managed objects (Tensor InsideDescribe,
        // Backend::MemObj, ...) get released concurrently from different threads,
        // so a plain int loses updates and double-frees. Do not "optimize" this
        // back to a non-atomic counter — a plain optimized build cannot catch the
        // regression at runtime (the compiler elides the racy RMW pair as UB),
        // which is why this guard is compile-time.
        // See test/core/RefCountThreadSafetyTest.cpp.
        static_assert(std::is_same<decltype(mNum), std::atomic<int>>::value,
                      "RefCount::mNum must remain std::atomic<int>");
};

#define SAFE_UNREF(x)\
    if (NULL!=(x)) {(x)->decRef();}
#define SAFE_REF(x)\
    if (NULL!=(x)) (x)->addRef();

#define SAFE_ASSIGN(dst, src) \
    {\
        if (src!=NULL)\
        {\
            src->addRef();\
        }\
        if (dst!=NULL)\
        {\
            dst->decRef();\
        }\
        dst = src;\
    }
template <typename T>
class SharedPtr {
    public:
        SharedPtr() : mT(NULL) {}
        SharedPtr(T* obj) : mT(obj) {}
        SharedPtr(const SharedPtr& o) : mT(o.mT) { SAFE_REF(mT); }
        ~SharedPtr() { SAFE_UNREF(mT); }

        SharedPtr& operator=(const SharedPtr& rp) {
            SAFE_ASSIGN(mT, rp.mT);
            return *this;
        }
        SharedPtr& operator=(T* obj) {
            SAFE_UNREF(mT);
            mT = obj;
            return *this;
        }

        T* get() const { return mT; }
        T& operator*() const { return *mT; }
        T* operator->() const { return mT; }

    private:
        T* mT;
};

struct BufferStorage {
    size_t size() const {
        return allocated_size - offset;
    }

    const uint8_t* buffer() const {
        return storage + offset;
    }
    ~ BufferStorage() {
        if (nullptr != storage) {
            delete [] storage;
        }
    }
    size_t allocated_size;
    size_t offset;
    uint8_t* storage = nullptr;
};

} // namespace MNN

#endif /* AutoStorage_h */
