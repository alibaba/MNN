//
//  BufferAllocator.hpp
//  MNN
//
//  Created by MNN on 2018/08/20.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef BufferAllocator_hpp
#define BufferAllocator_hpp

#include <map>
#include <memory>
#include <vector>
#include "MNNMemoryUtils.h"
#include "NonCopyable.hpp"

namespace MNN {

/** memory utils wrapper. provides memory reusing with alignment ability. */
class MNN_PUBLIC BufferAllocator : public NonCopyable {
public:
    /**
     * @brief init buffer allocator with pointer alignment.
     * @param align given pointer alignment.
     */
    BufferAllocator(int align = MNN_MEMORY_ALIGN_DEFAULT) : mAlign(align) {
        // nothing to do
    }
    /**
     * @brief deinit buffer allocator. frees all allocated memories.
     */
    ~BufferAllocator() {
        release();
    }

public:
    /**
     * @brief alloc CHUNK pointer with given size. if any reusable pointer matches size, reuse it.
     * @param size  given size.
     * @param seperate if true, the memory can't be alloc from free pool
     * @return allocated or used CHUNK pointer.
     * @sa free
     * @sa release
     */
    void* alloc(size_t size, bool seperate = false);

    /**
     * @brief mark CHUNK pointer as reusable.
     * @param pointer   given CHUNK pointer.
     * @param release   true if need free directly.
     * @return true if pointer is a CHUNK pointer, false otherwise.
     * @sa release
     */
    bool free(void* pointer, bool release = false);

    /**
     * @brief free all allocated memories.
     * @sa allocSeparate
     * @sa alloc
     * if allRelease, clear all memory , otherwise delete freelist
     */
    void release(bool allRelease = true);

    /**
     * @brief query total size allocated indeed.
     * @return total size allocated indeed.
     */
    size_t totalSize() const {
        return mTotalSize;
    }

    /*
     For multi thread case,
     we must assume that the memory use by different thread don't conflict
     begin barrier / end barrier means enter the alloc for multi-thread
     begin group / end group means the memory allocated belong to one thread
     different group must use different memory,
     but the origin freelist can be used by every group
     */
    void barrierBegin();
    void barrierEnd();
    void beginGroup();
    void endGroup();

private:
    class Node {
    public:
        ~Node();
        void* pointer;
        size_t size;
        std::shared_ptr<Node> parent = nullptr;
        int useCount                 = 0;
    };

    typedef std::multimap<size_t, std::shared_ptr<Node>> FREELIST;

    static void returnMemory(FREELIST* list, std::shared_ptr<Node> node, bool permitMerge = true);
    void* getFromFreeList(FREELIST* list, size_t size, bool permiteSplit = true);

    std::map<void*, std::shared_ptr<Node>> mUsedList;
    FREELIST mFreeList;
    size_t mTotalSize   = 0;
    const size_t mAlign = 0;

    FREELIST* mCurrenetFreeList = nullptr;
    std::vector<std::shared_ptr<FREELIST>> mGroups;
};
} // namespace MNN
#endif
