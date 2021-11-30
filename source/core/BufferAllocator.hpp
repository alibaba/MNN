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
#include "AutoStorage.h"

namespace MNN {

/** memory utils wrapper. provides memory reusing with alignment ability. */
class MNN_PUBLIC BufferAllocator : public NonCopyable {
public:
    class Allocator {
    public:
        Allocator() = default;
        virtual ~ Allocator() = default;
        virtual std::pair<void*, int> onAlloc(int size, int align) = 0;
        virtual void onRelease(std::pair<void*, int> ptr) = 0;
        static std::shared_ptr<Allocator> createDefault();
        static std::shared_ptr<Allocator> createRecurse(BufferAllocator* parent);
    };
    /**
     * @brief init buffer allocator with pointer alignment.
     * @param align given pointer alignment.
     */
    BufferAllocator(std::shared_ptr<Allocator> parent, int align = MNN_MEMORY_ALIGN_DEFAULT) : mAllocator(parent), mAlign(align) {
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
    std::pair<void*, int> alloc(int size, bool seperate = false, int align = 0);

    /**
     * @brief mark CHUNK pointer as reusable.
     * @param pointer   given CHUNK pointer.
     * @return true if pointer is a CHUNK pointer, false otherwise.
     * @sa release
     */
    bool free(std::pair<void*, int> pointer);

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
    class Node : public RefCount {
    public:
        ~Node();
        std::pair<void*, int> pointer;
        SharedPtr<Node> parent = nullptr;
        int32_t size;
        int16_t useCount = 0;
        Allocator* outside = nullptr;
    };

    typedef std::multimap<size_t, SharedPtr<Node>> FREELIST;

    static void returnMemory(FREELIST* list, SharedPtr<Node> node, bool permitMerge = true);
    std::pair<void*, int> getFromFreeList(FREELIST* list, int size, bool permiteSplit, int align);

    std::map<std::pair<void*, int>, SharedPtr<Node>> mUsedList;
    FREELIST mFreeList;
    size_t mTotalSize   = 0;

    FREELIST* mCurrentFreeList = nullptr;
    std::vector<std::shared_ptr<FREELIST>> mGroups;
    std::shared_ptr<Allocator> mAllocator;
    int mAlign;
};
} // namespace MNN
#endif
