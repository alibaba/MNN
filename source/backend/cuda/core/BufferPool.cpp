//
//  BufferPool.cpp
//  MNN
//
//  Created by MNN on 2018/12/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "BufferPool.hpp"
//#define DUMP_USAGE
//#define MNN_DEBUG_MEMORY
namespace MNN {
namespace CUDA {
BufferPool::Node::~Node() {
    if (nullptr == parent) {
        runtime->free(pointer);
    }
}
void* BufferPool::alloc(size_t size, bool seperate) {
#ifdef DUMP_USAGE
    auto memoryUsed = size / 1024.0f / 1024.0f;
    MNN_PRINT("Alloc: %f\n", memoryUsed);
#endif
    void* pointer = nullptr;
    // reuse if possible
    if (!seperate) {
        pointer = getFromFreeList(&mFreeList, size);
        if (nullptr != pointer) {
            return pointer;
        }
    }

    // alloc otherwise
    pointer = mRuntime->alloc(size);
    if (nullptr == pointer) {
        return nullptr;
    }
    mTotalSize += size;

    // save node
    std::shared_ptr<Node> node(new Node);
    node->size         = size;
    node->pointer      = pointer;
    node->runtime      = mRuntime;
    mUsedList[pointer] = node;

#ifdef DUMP_USAGE
    MNN_PRINT("mTotalSize: %f\n", mTotalSize / 1024.0f / 1024.0f);
#endif
    return pointer;
}

void BufferPool::returnMemory(FREELIST* listP, std::shared_ptr<Node> node, bool permitMerge) {
    auto& list = *listP;
    list.insert(std::make_pair(node->size, node));
    // update parent use count
    if (nullptr != node->parent && permitMerge) {
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

bool BufferPool::free(void* pointer, bool needRelease) {
    // get node
    auto x = mUsedList.find(pointer);
    if (x == mUsedList.end()) {
        MNN_ASSERT(false);
        return false;
    }
    if (needRelease) {
        MNN_ASSERT(x->second->parent == nullptr);
        mTotalSize -= x->second->size;
        mUsedList.erase(x);
        return true;
    }

    // mark as reusable
    auto node = x->second;
    mUsedList.erase(x);
    returnMemory(&mFreeList, node);
#ifdef DUMP_USAGE
    auto memoryUsed = x->second->size / 1024.0f / 1024.0f;
    MNN_PRINT("Free: %f\n", memoryUsed);
#endif
    return true;
}

void BufferPool::release(bool allRelease) {
    if (allRelease) {
        mUsedList.clear();
        mFreeList.clear();
        mTotalSize = 0;
        return;
    }
    for (auto f : mFreeList) {
        mTotalSize -= f.first;
    }
    mFreeList.clear();
}

void* BufferPool::getFromFreeList(FREELIST* list, size_t size, bool permiteSplit) {
#ifdef MNN_DEBUG_MEMORY
    return nullptr;
#endif

    // get node larger than size
    auto x = list->lower_bound(size);
    if (x == list->end()) {
        return nullptr;
    }

    // update parent use count
    void* pointer = x->second->pointer;
    if (permiteSplit && nullptr != x->second->parent) {
        x->second->parent->useCount += 1;
    }

    // uses up all aligned space
    auto sizeAlign = size;
    if (sizeAlign >= x->first || (!permiteSplit)) {
        mUsedList.insert(std::make_pair(pointer, x->second));
        list->erase(x);
        return pointer;
    }

    // split otherwise
    std::shared_ptr<Node> first(new Node);
    first->parent  = x->second;
    first->size    = sizeAlign;
    first->pointer = x->second->pointer;
    mUsedList.insert(std::make_pair(pointer, first));
    x->second->useCount += 1;

    std::shared_ptr<Node> second(new Node);
    second->parent  = x->second;
    second->size    = x->second->size - sizeAlign;
    second->pointer = ((uint8_t*)x->second->pointer) + sizeAlign;
    list->insert(std::make_pair(second->size, second));
    list->erase(x);
    return pointer;
}
} // namespace CUDA
} // namespace MNN