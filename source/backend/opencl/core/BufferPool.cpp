//
//  BufferPool.cpp
//  MNN
//
//  Created by MNN on 2019/02/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "BufferPool.hpp"
namespace MNN {
namespace OpenCL {
cl::Buffer* BufferPool::alloc(int size, bool seperate) {
    if (!seperate) {
        auto iter = mFreeList.lower_bound(size);
        if (iter != mFreeList.end()) {
            auto buffer = iter->second->buffer.get();
            mFreeList.erase(iter);
            return buffer;
        }
    }
    std::shared_ptr<Node> node(new Node);
    node->size = size;
    node->buffer.reset(new cl::Buffer(mContext, mFlag, size));
    mAllBuffer.insert(std::make_pair(node->buffer.get(), node));

    return node->buffer.get();
}

void BufferPool::recycle(cl::Buffer* buffer, bool release) {
    auto iter = mAllBuffer.find(buffer);
    if (iter == mAllBuffer.end()) {
        MNN_ERROR("Error for recycle buffer\n");
        return;
    }
    if (release) {
        mAllBuffer.erase(iter);
        return;
    }
    mFreeList.insert(std::make_pair(iter->second->size, iter->second));
}

void BufferPool::clear() {
    mFreeList.clear();
    mAllBuffer.clear();
}
} // namespace OpenCL
} // namespace MNN
