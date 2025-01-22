//
//  BufferAllocator.cpp
//  MNN
//
//  Created by MNN on 2018/12/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <string>
#include "core/BufferAllocator.hpp"
#include "core/Macro.h"
#include "MNNFileUtils.h"

// #define DUMP_USAGE
//#define MNN_DEBUG_MEMORY
namespace MNN {
// MemChunk function
bool MemChunk::invalid() const {
    return mNode == nullptr && first == nullptr;
}
void* MemChunk::base() const {
    if (mNode) {
        return mNode->base;
    }
    return first;
}
MemChunk MemChunk::operator+ (size_t offset) {
    auto chunk = *this;
    chunk.second += offset;
    return chunk;
}
size_t MemChunk::offset() const {
    if (mNode) {
        return mNode->offset + second;
    }
    return second;
}
void MemChunk::attach(Tensor* tensor) {
    if (mNode) {
        mNode->tensors.push_back(tensor);
    }
}
ErrorCode BufferAllocator::compute() {
    return NO_ERROR;
}
ErrorCode BufferAllocator::apply() {
    return NO_ERROR;
}

class DefaultAllocator : public BufferAllocator::Allocator {
public:
    DefaultAllocator() {
        // Do nothing
    }
    virtual ~ DefaultAllocator() {
        // Do nothing
    }
    virtual MemChunk onAlloc(size_t size, size_t align) {
        return MemChunk(MNNMemoryAllocAlign(size, MNN_MEMORY_ALIGN_DEFAULT), 0);
    }
    virtual void onRelease(MemChunk chunk) {
        MNN_ASSERT(chunk.second == 0);
        MNNMemoryFreeAlign(chunk.first);
    }
};
class MmapAllocator : public BufferAllocator::Allocator {
private:
    std::map<void*, std::tuple<file_t,size_t, std::string>> mCache;
    std::string mFileName;
    std::string mPrefix;
    std::string mPosfix;
    int mAllocTimes = 0;
    bool mRemove;
    bool mNewMmap = false;

public:
    MmapAllocator(const char* dirName, const char* prefix, const char* posfix, bool autoRemove) {
        if (nullptr != dirName) {
            mFileName = dirName;
            if (!MNNCreateDir(dirName)) {
                MNN_ERROR("%s not exist\n", dirName);
            }
        }
        if (nullptr != prefix) {
            mPrefix = prefix;
        }
        if (nullptr != posfix) {
            mPosfix = posfix;
        }
        mRemove = autoRemove;
    }
    virtual ~ MmapAllocator() {
        for (auto& iter : mCache) {
            MNNUnmapFile(iter.first, std::get<1>(iter.second));
            MNNCloseFile(std::get<0>(iter.second));
            if (mRemove) {
                MNNRemoveFile(std::get<2>(iter.second).c_str());
            }
        }
    }
    virtual MemChunk onAlloc(size_t size, size_t align) {
        MNN_ASSERT(size > 0);
        std::string name = mPrefix + std::to_string(mAllocTimes) + "." + mPosfix;
        std::string fileName = MNNFilePathConcat(mFileName, name);
        file_t file;
        if (MNNFileExist(fileName.c_str())) {
            file = MNNOpenFile(fileName.c_str(), MNN_FILE_READ | MNN_FILE_WRITE);
        } else {
            file = MNNCreateFile(fileName.c_str());
            size = UP_DIV(size, align) * align;
            auto code = MNNSetFileSize(file, size);
            if (NO_ERROR != code) {
                MNN_ERROR("Set File size %lu error= %d\n", size, code);
            }
            mNewMmap = true;
        }
        void* ptr = MNNMmapFile(file, size);
        mCache.insert(std::make_pair(ptr, std::make_tuple(file, size, fileName)));
        mAllocTimes++;
        return MemChunk(ptr, 0);
    }
    virtual void onRelease(MemChunk chunk) override {
        MNN_ASSERT(chunk.second == 0);
        auto iter = mCache.find(chunk.first);
        if (iter == mCache.end()) {
            MNN_ASSERT(false);
            MNN_ERROR("Invalid free for MMAPAllocator\n");
            return;
        }
        MNNUnmapFile(iter->first, std::get<1>(iter->second));
        MNNCloseFile(std::get<0>(iter->second));
        if (mRemove) {
            MNNRemoveFile(std::get<2>(iter->second).c_str());
        }
        mCache.erase(iter);
        mAllocTimes = 0;
    }
    virtual void sync() override {
        if (!mRemove && mNewMmap) {
            for (auto& iter : mCache) {
                MNNMmapSync(iter.first, std::get<1>(iter.second));
            }
            std::string cacheName = mPrefix + "sync." + mPosfix;
            std::string fileName = MNNFilePathConcat(mFileName, cacheName);
            MNNCreateFile(fileName.c_str());
        }
    }
};
class RecurseAllocator : public BufferAllocator::Allocator {
public:
    RecurseAllocator(BufferAllocator* parent) {
        mParent = parent;
    }
    virtual ~ RecurseAllocator() {
        // Do nothing
    }
    virtual MemChunk onAlloc(size_t size, size_t align) override {
        return mParent->alloc(size, false, align);
    }
    virtual void onRelease(MemChunk chunk) override {
        mParent->free(chunk);
    }
private:
    BufferAllocator* mParent;
};

std::shared_ptr<BufferAllocator::Allocator> BufferAllocator::Allocator::createDefault() {
    std::shared_ptr<BufferAllocator::Allocator> _res;
    _res.reset(new DefaultAllocator);
    return _res;
}
std::shared_ptr<BufferAllocator::Allocator> BufferAllocator::Allocator::createMmap(const char* dirName, const char* prefix, const char* posfix, bool autoRemove) {
    std::shared_ptr<BufferAllocator::Allocator> _res;
    _res.reset(new MmapAllocator(dirName, prefix, posfix, autoRemove));
    return _res;
}

std::shared_ptr<BufferAllocator::Allocator> BufferAllocator::Allocator::createRecurse(BufferAllocator* parent) {
    std::shared_ptr<BufferAllocator::Allocator> _res;
    _res.reset(new RecurseAllocator(parent));
    return _res;
}

EagerBufferAllocator::Node::~Node() {
    if (nullptr == parent.get()) {
        outside->onRelease(pointer);
    }
}
MemChunk EagerBufferAllocator::alloc(size_t size, bool separate, size_t align) {
#ifdef DUMP_USAGE
    auto memoryUsed = size / 1024.0f / 1024.0f;
    MNN_PRINT("Alloc: %f\n", memoryUsed);
#endif
    if (0 == align) {
        align = mAlign;
    }
    std::pair<void*, size_t> pointer;
    // reuse if possible
    if (!separate) {
        if (nullptr != mCurrentFreeList) {
            pointer = getFromFreeList(mCurrentFreeList, size, false, align);
        }
        if (nullptr != pointer.first) {
            return MemChunk(pointer);
        }
        pointer = getFromFreeList(&mFreeList, size, true, align);
        if (nullptr != pointer.first) {
            return MemChunk(pointer);
        }
    }
    auto allocSize = size;
    if (mMinAllocSize != 0) {
        allocSize = ALIMAX(mMinAllocSize, size);
    }

    // alloc otherwise
    auto chunk = mAllocator->onAlloc(allocSize, align);
    pointer.first = chunk.first;
    pointer.second = chunk.second;
    if (nullptr == pointer.first) {
        return chunk;
    }
    mTotalSize += allocSize;

    // save node
    SharedPtr<Node> node(new Node);
    node->size         = allocSize;
    node->pointer      = pointer;
    node->outside      = mAllocator.get();
    MNN_ASSERT(pointer.second % align == 0);
    if (allocSize > size) {
        // Split
        SharedPtr<Node> first(new Node);
        first->parent  = node;
        first->size    = size;
        first->pointer = pointer;
        mUsedList.insert(std::make_pair(pointer, first));
        node->useCount = 1;

        SharedPtr<Node> second(new Node);
        second->parent  = node;
        second->size    = allocSize - size;
        second->pointer.first = pointer.first;
        second->pointer.second = pointer.second + size;
        if (nullptr != mCurrentFreeList) {
            mCurrentFreeList->insert(std::make_pair(second->size, second));
        } else {
            mFreeList.insert(std::make_pair(second->size, second));
        }
    } else {
        mUsedList[pointer] = node;
    }
#ifdef DUMP_USAGE
    MNN_PRINT("mTotalSize: %f\n", mTotalSize / 1024.0f / 1024.0f);
#endif
    return pointer;
}

void EagerBufferAllocator::returnMemory(FREELIST* listP, SharedPtr<Node> node, bool permitMerge) {
    auto& list = *listP;
    list.insert(std::make_pair(node->size, node));
    // update parent use count
    if (nullptr != node->parent.get() && permitMerge) {
        auto parent = node->parent;
        parent->useCount -= 1;

        // merge if all subnodes were freed
        auto needMerge = parent->useCount == 0;
        while (needMerge) {
            // collect all subnodes
            for (auto iter = list.begin(); iter != list.end();) {
                if (iter->second->parent.get() == parent.get()) {
                    iter = list.erase(iter);
                    continue;
                }
                iter++;
            }

            // do merge downside up
            list.insert(std::make_pair(parent->size, parent));
            needMerge = false;
            if (parent->parent.get() != nullptr) {
                parent = parent->parent;
                parent->useCount -= 1;
                needMerge = parent->useCount == 0;
            }
        }
    }
}

bool EagerBufferAllocator::free(MemChunk chunk) {
    std::pair<void*, size_t> pointer(chunk.first, chunk.second);
    // get node
    auto x = mUsedList.find(pointer);
    if (x == mUsedList.end()) {
        MNN_ASSERT(false);
        return false;
    }
    // mark as reusable
    auto node = x->second;
    mUsedList.erase(x);
    if (nullptr != mCurrentFreeList) {
        returnMemory(mCurrentFreeList, node, false);
    } else {
        returnMemory(&mFreeList, node);
    }
#ifdef DUMP_USAGE
    if (node.get()) {
        auto memoryUsed = node->size / 1024.0f / 1024.0f;
        MNN_PRINT("Free: %f\n", memoryUsed);
    }
#endif
    return true;
}

void EagerBufferAllocator::release(bool allRelease) {    
    MNN_ASSERT(mGroups.empty());
    if (allRelease) {
        mUsedList.clear();
        mFreeList.clear();
        mTotalSize = 0;
        return;
    }
    for (auto f : mFreeList) {
        if (f.second->parent.get() == nullptr) {
            MNN_ASSERT(mTotalSize >= f.first);
            mTotalSize -= f.first;
        }
    }
    mFreeList.clear();
}

void EagerBufferAllocator::barrierBegin() {
    MNN_ASSERT(mGroups.empty());
}

void EagerBufferAllocator::barrierEnd() {
    for (auto& freeGroup : mGroups) {
        auto freeList = *freeGroup;
        for (auto& iter : freeList) {
            returnMemory(&mFreeList, iter.second);
        }
    }
    mGroups.clear();
}

void EagerBufferAllocator::beginGroup() {
    std::shared_ptr<FREELIST> newFreeList(new FREELIST);
    mCurrentFreeList = newFreeList.get();
    mGroups.emplace_back(newFreeList);
}

void EagerBufferAllocator::endGroup() {
    mCurrentFreeList = nullptr;
}

void EagerBufferAllocator::sync() {
    mAllocator->sync();
}

std::pair<void*, size_t> EagerBufferAllocator::getFromFreeList(FREELIST* list, size_t size, bool permiteSplit, size_t align) {
#ifdef MNN_DEBUG_MEMORY
    return std::make_pair(nullptr, 0);
#endif
    size_t realSize = size;
    bool needExtraSize = mAlign % align != 0;
    if (needExtraSize) {
        realSize = size + align - 1;
    }
    // get node larger than size
    auto x = list->lower_bound(realSize);
    if (x == list->end()) {
        return std::make_pair(nullptr, 0);
    }
    // update parent use count
    auto pointer = x->second->pointer;
    // Align offset
    if (needExtraSize) {
        size_t originOffset = pointer.second;
        pointer.second = UP_DIV(originOffset, align) * align;
        realSize = size + pointer.second - originOffset;
    }
    if (permiteSplit && nullptr != x->second->parent.get()) {
        x->second->parent->useCount += 1;
    }

    // uses up all aligned space
    auto sizeAlign = UP_DIV(realSize, mAlign) * mAlign;
    if (sizeAlign >= x->first || (!permiteSplit)) {
        mUsedList.insert(std::make_pair(pointer, x->second));
        list->erase(x);
        MNN_ASSERT(pointer.second % align == 0);
        return pointer;
    }

    // split otherwise
    SharedPtr<Node> first(new Node);
    first->parent  = x->second;
    first->size    = sizeAlign;
    first->pointer = x->second->pointer;
    mUsedList.insert(std::make_pair(pointer, first));
    x->second->useCount += 1;

    SharedPtr<Node> second(new Node);
    second->parent  = x->second;
    second->size    = x->second->size - sizeAlign;
    second->pointer.first = x->second->pointer.first;
    second->pointer.second = x->second->pointer.second + sizeAlign;
    list->erase(x);
    list->insert(std::make_pair(second->size, second));
    MNN_ASSERT(pointer.second % align == 0);
    return pointer;
}
static void _CPUMemChunkApplyToTensor(uint8_t* ptr, size_t offset, Tensor* t) {
    t->buffer().host = ptr + offset;
}
SingleBufferWithAllocator::~ SingleBufferWithAllocator() {
    release();
}
void SingleBufferWithAllocator::release() {
    if (current.first != nullptr) {
        root->onRelease(current);
        current.first = nullptr;
        current.second = 0;
        currentSize = 0;
    }
}

ErrorCode SingleBufferWithAllocator::realloc(size_t size, size_t align) {
    if (currentSize < size) {
        if (nullptr != current.first) {
            root->onRelease(current);
        }
        current = root->onAlloc(size, align);
        if (current.first == nullptr) {
            return OUT_OF_MEMORY;
        }
        currentSize = size;
    }
    return NO_ERROR;
}


DeferBufferAllocator::DeferBufferAllocator(SingleBufferWithAllocator* root, size_t align, MemChunkApplyToTensor func) : mAlign(align) {
    if (nullptr == func) {
        mApplyFunction = _CPUMemChunkApplyToTensor;
    } else {
        mApplyFunction = func;
    }
    mParent = root;
}

//------------------------------- DeferBufferAllocator -----------------------------------//
MemChunk DeferBufferAllocator::alloc(size_t size, bool separate, size_t align) {
    if (0 == align) {
        align = mAlign;
    }
    size = UP_DIV(size, align) * align;
    if (mFreeList.empty() || separate) {
        auto newChunk = createMemNode(size);
        insert_after(newChunk);
#ifdef DUMP_USAGE
    MNN_PRINT("Defer alloc: %p, %d\n", newChunk, size);
#endif
        return MemChunk(newChunk);
    }
    std::unique_ptr<MemNode> tmpChunk(new MemNode(size));
    auto iter = mFreeList.lower_bound(ChunkBySize(tmpChunk.get()));
    if (iter == mFreeList.end()) {
        --iter;
    }
    auto selectChunk = iter->chunk;
    mFreeList.erase(iter);
    selectChunk->usage = true;
    if (selectChunk->size > size) {
        // split `[####]` to `[###]->[#]`
        auto restChunk = createMemNode(selectChunk->size - size);
        restChunk->usage = false;
        insert_after(restChunk, selectChunk);
        // add `[#]` to freelist
        insertFree(restChunk);
    }
    // equal no change; small expand
    selectChunk->size = size;
#ifdef DUMP_USAGE
    MNN_PRINT("Defer alloc: %p, %d\n", selectChunk, size);
#endif
    return MemChunk(selectChunk);
}
bool DeferBufferAllocator::free(MemChunk chunk) {
#ifdef DUMP_USAGE
    MNN_PRINT("Defer free: %p\n", chunk.mNode);
#endif
    if (mBarrrier) {
        mBarrrierFreeChunks.emplace_back(std::move(chunk));
        return true;
    }
    auto node = chunk.mNode;
    if (!node) {
        return false;
    }
    auto left = node->left;
    auto right = node->right;
    if (left && !left->usage) {
        // fuse to left
        eraseFree(left);
        node = fuse_to_left(left, node);
    }
    if (right && !right->usage) {
        // fuse to left
        eraseFree(right);
        node = fuse_to_left(node, right);
    }
    node->usage = false;
    insertFree(node);
    return true;
}

void DeferBufferAllocator::release(bool allRelease) {
    if (allRelease) {
        reset();
    }
}

void DeferBufferAllocator::barrierBegin() {
    MNN_ASSERT(!mBarrrier);
    mBarrrier = true;
}
void DeferBufferAllocator::barrierEnd() {
    mBarrrier = false;
    MNN_ASSERT(!mBarrrier);
    for (auto& chunk : mBarrrierFreeChunks) {
        this->free(chunk);
    }
    mBarrrierFreeChunks.clear();
}
void DeferBufferAllocator::beginGroup() {
    // do nothing
}
void DeferBufferAllocator::endGroup() {
    // do nothing
}

void DeferBufferAllocator::reset() {
    mTotalSize = 0;
    mChunks.clear();
    mFreeList.clear();
    mPtr.first = nullptr;
    mPtr.second = 0;
    mHead = nullptr;
    mTail = nullptr;
    mBarrrier = false;
    mBarrrierFreeChunks.clear();
}

ErrorCode DeferBufferAllocator::compute() {
    if (mTotalSize > 0) {
        return NO_ERROR;
    }
    mTotalSize = 0;
    if (mFreeList.empty()) {
        return NO_ERROR;
    }
    MNN_ASSERT(mFreeList.size() == 1);
    MNN_ASSERT(mHead == mTail);
    if (mFreeList.size() != 1 || mHead != mTail) {
        // Defer allocator compute error
        return INVALID_VALUE;
    }
    auto chunk = mHead;
    while (chunk) {
        chunk->offset = mTotalSize;
        visiChildren(chunk);
        mTotalSize += chunk->size;
        chunk = chunk->right;
    }
    return apply();
}
ErrorCode DeferBufferAllocator::apply() {
    if (mFreeList.empty()) {
        // Not alloc
        return NO_ERROR;
    }
    auto& chunk = mParent->current;
    bool needApply = false;
    if (mParent->currentSize < mTotalSize) {
        needApply = true;
        auto code = mParent->realloc(mTotalSize, mAlign);
        if (NO_ERROR != code) {
            return code;
        }
    } else if (mPtr.first != chunk.first || mPtr.second != chunk.second) {
        needApply = true;
    }
    if (!needApply) {
        return NO_ERROR;
    }
    mPtr = chunk;
    for (auto& chunk : mChunks) {
        chunk->base = mPtr.ptr();
        for (auto t : chunk->tensors) {
            mApplyFunction((uint8_t*)mPtr.base(), chunk->offset + mPtr.offset(), t);
        }
    }
    return NO_ERROR;
}

// some utils functions of DeferBufferAllocator
void DeferBufferAllocator::visiChildren(MemNode* chunk) {
    if (!chunk) return;
    for (auto child : chunk->children) {
        child->offset += chunk->offset;
        visiChildren(child);
    }
}
MemNode* DeferBufferAllocator::fuse_to_left(MemNode* left, MemNode* right) {
    right->offset = left->size;
    left->size += right->size;
    left->children.push_back(right);
    erase_node(right);
    return left;
}
void DeferBufferAllocator::erase_node(MemNode* chunk) {
    auto left = chunk->left;
    auto right = chunk->right;
    if (left && right) {
        left->right = right;
        right->left = left;
        return;
    }
    if (left) {
        left->right = nullptr;
        mTail = left;
        return;
    }
    if (right) {
        right->left = nullptr;
        mHead = right;
        return;
    }
    mHead = mTail = nullptr;
}
void DeferBufferAllocator::insert_after(MemNode* chunk, MemNode* pos) {
    if (pos) {
        auto right = pos->right;
        if (right) {
            right->left = chunk;
        }
        chunk->right = right;
        chunk->left = pos;
        pos->right = chunk;
        if (pos == mTail) {
            mTail = chunk;
        }
    } else if (mTail) {
        mTail->right = chunk;
        chunk->left = mTail;
        mTail = chunk;
    } else {
        mHead = chunk;
        mTail = chunk;
    }
}
MemNode* DeferBufferAllocator::createMemNode(size_t size) {
    mChunks.emplace_back(new MemNode(size));
    return mChunks.back().get();
}
void DeferBufferAllocator::insertFree(MemNode* chunk) {
    mFreeList.insert(ChunkBySize(chunk));
}
void DeferBufferAllocator::eraseFree(MemNode* chunk) {
    auto range = mFreeList.equal_range(ChunkBySize(chunk));
    for (auto iter = range.first; iter != range.second; iter++) {
        if (iter->chunk == chunk) {
            mFreeList.erase(iter);
            break;
        }
    }
}

} // namespace MNN
