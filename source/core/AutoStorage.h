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
        if (NULL != mData) {
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
        if (NULL != mData) {
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
            mNum++;
        }
        void decRef() const
        {
            --mNum;
            MNN_ASSERT(mNum>=0);
            if (0 >= mNum)
            {
                delete this;
            }
        }
    inline int count() const{return mNum;}
    protected:
        RefCount():mNum(1){}
        RefCount(const RefCount& f):mNum(f.mNum){}
        void operator=(const RefCount& f)
        {
            if (this != &f)
            {
                mNum = f.mNum;
            }
        }
        virtual ~RefCount(){}
    private:
        mutable int mNum;
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
