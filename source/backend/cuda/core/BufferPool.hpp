//
//  BufferPool.hpp
//  MNN
//
//  Created by MNN on 2019/02/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef BufferPool_hpp
#define BufferPool_hpp

#include <map>
#include <memory>
#include <vector>
#include "runtime/CUDARuntime.hpp"
namespace MNN {
namespace CUDA {
/** memory utils wrapper. provides memory reusing with alignment ability. */
class BufferPool {
public:
    /**
     * @brief init buffer allocator with pointer alignment.
     * @param CUDARuntime given runtime.
     */
    BufferPool(CUDARuntime* runtime) : mRuntime(runtime) {
        // nothing to do
    }
    /**
     * @brief deinit buffer allocator. frees all allocated memories.
     */
    ~BufferPool() {
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

private:
    class Node {
    public:
        ~Node();
        void* pointer;
        size_t size;
        std::shared_ptr<Node> parent = nullptr;
        int useCount                 = 0;
        CUDARuntime* runtime;
    };

    typedef std::multimap<size_t, std::shared_ptr<Node>> FREELIST;

    static void returnMemory(FREELIST* list, std::shared_ptr<Node> node, bool permitMerge = true);
    void* getFromFreeList(FREELIST* list, size_t size, bool permiteSplit = true);

    std::map<void*, std::shared_ptr<Node>> mUsedList;
    FREELIST mFreeList;
    size_t mTotalSize = 0;
    CUDARuntime* mRuntime;
};
} // namespace CUDA
} // namespace MNN
#endif
