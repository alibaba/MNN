//
//  BufferAllocator.cpp
//  MNN
//
//  Created by MNN on 2018/12/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "core/BufferAllocator.hpp"
#include "core/Macro.h"

//#define DUMP_USAGE
//#define MNN_DEBUG_MEMORY
namespace MNN {
BufferAllocator::Node::~Node() {
    if (nullptr == parent) {
        MNNMemoryFreeAlign(pointer);
    }
}
void* BufferAllocator::alloc(size_t size, bool separate) {
#ifdef DUMP_USAGE
    auto memoryUsed = size / 1024.0f / 1024.0f;
    MNN_PRINT("Alloc: %f\n", memoryUsed);
#endif
    void* pointer = nullptr;
    // reuse if possible
    if (!separate) {
        if (nullptr != mCurrentFreeList) {
            pointer = getFromFreeList(mCurrentFreeList, size, false);
        }
        if (nullptr != pointer) {
            return pointer;
        }
        pointer = getFromFreeList(&mFreeList, size);
        if (nullptr != pointer) {
            return pointer;
        }
    }

    // alloc otherwise
    pointer = MNNMemoryAllocAlign(size, mAlign);
    if (nullptr == pointer) {
        return nullptr;
    }
    mTotalSize += size;

    // save node
    std::shared_ptr<Node> node(new Node);
    node->size         = size;
    node->pointer      = pointer;
    mUsedList[pointer] = node;

#ifdef DUMP_USAGE
    MNN_PRINT("mTotalSize: %f\n", mTotalSize / 1024.0f / 1024.0f);
#endif
    return pointer;
}

void BufferAllocator::returnMemory(FREELIST* listP, std::shared_ptr<Node> node, bool permitMerge) {
    auto& list = *listP;
    list.emplace(node->size, node);
    // update parent use count
    auto parent = node->parent;
    if (parent && permitMerge) {
        parent->useCount--;

        // merge if all subnodes were freed
        auto needMerge = parent->useCount == 0;
        while (needMerge) {
            // collect all subnodes
            for (auto iter = list.begin(); iter != list.end();) {
                if (iter->second->parent == parent) {
                    iter = list.erase(iter);
                    continue;
                }
                ++iter;
            }

            // do merge downside up
            list.emplace(parent->size, parent);
            needMerge = false;
            if (parent->parent != nullptr) {
                parent = parent->parent;
                parent->useCount--;
                needMerge = parent->useCount == 0;
            }
        }
    }
}

bool BufferAllocator::free(void* pointer, bool needRelease) {
    // get node
    auto x = mUsedList.find(pointer);
    if (x == mUsedList.end()) {
        MNN_ASSERT(false)
        return false;
    }
    if (needRelease) {
        MNN_ASSERT(x->second->parent == nullptr)
        MNN_ASSERT(mTotalSize >= x->second->size)
        mTotalSize -= x->second->size;
        mUsedList.erase(x);
        return true;
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
    auto memoryUsed = x->second->size / 1024.0f / 1024.0f;
    MNN_PRINT("Free: %f\n", memoryUsed);
#endif
    return true;
}

void BufferAllocator::release(bool allRelease) {
    MNN_ASSERT(mGroups.empty())
    if (allRelease) {
        mUsedList.clear();
        mFreeList.clear();
        mTotalSize = 0;
        return;
    }
    for (const auto& f : mFreeList) {
        if (f.second->parent == nullptr) {
            MNN_ASSERT(mTotalSize >= f.first)
            mTotalSize -= f.first;
        }
    }
    mFreeList.clear();
}

void BufferAllocator::barrierBegin() const {
    MNN_ASSERT(mGroups.empty())
}

void BufferAllocator::barrierEnd() {
    for (auto& freeGroup : mGroups) {
        auto freeList = *freeGroup;
        for (auto& iter : freeList) {
            returnMemory(&mFreeList, iter.second);
        }
    }
    mGroups.clear();
}

void BufferAllocator::beginGroup() {
    std::shared_ptr<FREELIST> newFreeList(new FREELIST);
    mCurrentFreeList = newFreeList.get();
    mGroups.emplace_back(newFreeList);
}

void BufferAllocator::endGroup() {
    mCurrentFreeList = nullptr;
}

void* BufferAllocator::getFromFreeList(FREELIST* list, size_t size, bool permitSplit) {
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
    if (permitSplit && nullptr != x->second->parent) {
        x->second->parent->useCount += 1;
    }

    // uses up all aligned space
    auto sizeAlign = UP_DIV(size, mAlign) * mAlign;
    if (sizeAlign >= x->first || (!permitSplit)) {
        mUsedList.emplace(pointer, x->second);
        list->erase(x);
        return pointer;
    }

    // split otherwise
    std::shared_ptr<Node> first(new Node);
    first->parent  = x->second;
    first->size    = sizeAlign; // risky: convert from unsigned long long to int
    first->pointer = x->second->pointer;
    mUsedList.emplace(pointer, first);
    x->second->useCount++;

    std::shared_ptr<Node> second(new Node);
    second->parent  = x->second;
    second->size    = x->second->size - sizeAlign;
    second->pointer = static_cast<uint8_t*>(x->second->pointer) + sizeAlign;
    list->emplace(second->size, second);
    list->erase(x);
    return pointer;
}
} // namespace MNN
