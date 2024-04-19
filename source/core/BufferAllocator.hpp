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
#include <set>
#include <memory>
#include <vector>
#include "MNNMemoryUtils.h"
#include "NonCopyable.hpp"
#include "AutoStorage.h"
#include <MNN/Tensor.hpp>
#include <MNN/ErrorCode.hpp>

namespace MNN {

/** memory utils wrapper. provides memory reusing with alignment ability. */
class EagerBufferAllocator;
class DeferBufferAllocator;
class DefaultAllocator;

// some memory struct for allocator
struct MemNode {
public:
    MemNode(size_t s) : size(s) {}
    ~MemNode() {}
    size_t size = 0, offset = 0;
    void* base = nullptr;
    bool usage = true;
    MemNode *left = nullptr, *right = nullptr;
    std::vector<MemNode*> children;
    std::vector<Tensor*> tensors;
};

struct ChunkBySize {
public:
    ChunkBySize(MemNode* ch) : chunk(ch) {}
    MemNode* chunk;
    bool operator<(const ChunkBySize& rhs) const {
        return chunk->size < rhs.chunk->size;
    }
};

struct MemChunk {
public:
    MemChunk() = default;
    MemChunk(void* base, size_t offset = 0) : first(base), second(offset) {}
    MemChunk(std::pair<void*, size_t> pointer) : first(pointer.first), second(pointer.second) {}
    MemChunk(MemNode* node) : mNode(node) {}
    ~MemChunk() = default;
    MemChunk operator+ (size_t offset);
    void* base() const;
    size_t offset() const;
    bool invalid() const;
    void attach(Tensor* tensor);
    uint8_t* ptr() const {
        if (mNode) {
            return static_cast<uint8_t*>(mNode->base) + mNode->offset + second;
        }
        return static_cast<uint8_t*>(first) + second;
    }
public:
    void* first = nullptr;
    size_t second = 0;
private:
    MemNode* mNode = nullptr;
    friend class DeferBufferAllocator;
    friend class EagerBufferAllocator;
    friend class DefaultAllocator;
};

class MNN_PUBLIC BufferAllocator : public NonCopyable {
public:
    class Allocator {
    public:
        Allocator() = default;
        virtual ~ Allocator() = default;
        virtual MemChunk onAlloc(size_t size, size_t align) = 0;
        virtual void onRelease(MemChunk chunk) = 0;
        static std::shared_ptr<Allocator> createDefault();
        static std::shared_ptr<Allocator> createRecurse(BufferAllocator* parent);
    };
    BufferAllocator() = default;
    virtual ~BufferAllocator() = default;
    virtual MemChunk alloc(size_t size, bool separate = false, size_t align = 0) = 0;
    virtual bool free(MemChunk chunk) = 0;
    virtual void release(bool allRelease = true) = 0;
    virtual size_t totalSize() const = 0;
    virtual void barrierBegin() {}
    virtual void barrierEnd() {}
    virtual void beginGroup() {}
    virtual void endGroup() {}
    virtual void reset() {}
    virtual ErrorCode compute();
};


class MNN_PUBLIC EagerBufferAllocator : public BufferAllocator {
public:
    /**
     * @brief init buffer allocator with pointer alignment.
     * @param align given pointer alignment.
     */
    EagerBufferAllocator(std::shared_ptr<Allocator> parent, size_t align = MNN_MEMORY_ALIGN_DEFAULT) : mAllocator(parent), mAlign(align) {
        // nothing to do
    }
    /**
     * @brief deinit buffer allocator. frees all allocated memories.
     */
    ~EagerBufferAllocator() {
        release();
    }

public:
    /**
     * @brief alloc CHUNK pointer with given size. if any reusable pointer matches size, reuse it.
     * @param size  given size.
     * @param separate if true, the memory can't be alloc from free pool
     * @return allocated or used CHUNK pointer.
     * @sa free
     * @sa release
     */
    MemChunk alloc(size_t size, bool separate = false, size_t align = 0) override;

    /**
     * @brief mark CHUNK pointer as reusable.
     * @param pointer   given CHUNK pointer.
     * @return true if pointer is a CHUNK pointer, false otherwise.
     * @sa release
     */
    bool free(MemChunk chunk) override;

    /**
     * @brief free all allocated memories.
     * @sa allocSeparate
     * @sa alloc
     * if allRelease, clear all memory , otherwise delete freelist
     */
    void release(bool allRelease = true) override;

    /**
     * @brief query total size allocated indeed.
     * @return total size allocated indeed.
     */
    size_t totalSize() const override {
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
    void barrierBegin() override;
    void barrierEnd() override;
    void beginGroup() override;
    void endGroup() override;

private:
    class Node : public RefCount {
    public:
        ~Node();
        std::pair<void*, size_t> pointer;
        SharedPtr<Node> parent = nullptr;
        size_t size;
        size_t useCount = 0;
        Allocator* outside = nullptr;
    };

    typedef std::multimap<size_t, SharedPtr<Node>> FREELIST;

    static void returnMemory(FREELIST* list, SharedPtr<Node> node, bool permitMerge = true);
    std::pair<void*, size_t> getFromFreeList(FREELIST* list, size_t size, bool permiteSplit, size_t align);

    std::map<std::pair<void*, size_t>, SharedPtr<Node>> mUsedList;
    FREELIST mFreeList;
    size_t mTotalSize   = 0;

    FREELIST* mCurrentFreeList = nullptr;
    std::vector<std::shared_ptr<FREELIST>> mGroups;
    std::shared_ptr<Allocator> mAllocator;
    size_t mAlign;
};
typedef void(*MemChunkApplyToTensor)(uint8_t* ptr, size_t offset, Tensor* tensor);

class MNN_PUBLIC DeferBufferAllocator : public BufferAllocator {
public:
    DeferBufferAllocator(std::shared_ptr<Allocator> parent, size_t align = MNN_MEMORY_ALIGN_DEFAULT, MemChunkApplyToTensor func = nullptr);
    ~DeferBufferAllocator() {
        reset();
    }
public:
    MemChunk alloc(size_t size, bool separate = false, size_t align = 0) override;
    bool free(MemChunk chunk) override;
    void release(bool allRelease = true) override;
    size_t totalSize() const override;
    void barrierBegin() override;
    void barrierEnd() override;
    void beginGroup() override;
    void endGroup() override;
    void reset() override;
    ErrorCode compute() override;
private:
    std::vector<std::unique_ptr<MemNode>> mChunks;
    MemNode *mHead = nullptr, *mTail = nullptr;
    std::multiset<ChunkBySize> mFreeList;
    // std::unique_ptr<uint8_t[]> mPtr;
    MemChunk mPtr;
    size_t mTotalSize = 0;
    std::shared_ptr<Allocator> mAllocator;
    size_t mAlign;
    // barrier
    bool mBarrrier = false;
    std::vector<MemChunk> mBarrrierFreeChunks;
private:
    MemNode* createMemNode(size_t size);
    MemNode* fuse_to_left(MemNode* left, MemNode* right);
    void erase_node(MemNode* chunk);
    void insert_after(MemNode* chunk, MemNode* pos = nullptr);
    void insertFree(MemNode* chunk);
    void eraseFree(MemNode* chunk);
    void visiChildren(MemNode* chunk);
    MemChunkApplyToTensor mApplyFunction;
};
} // namespace MNN
#endif
