//
//  BufferPool.cpp
//  MNN
//
//  Created by MNN on 2019/02/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/opencl/core/BufferPool.hpp"
namespace MNN {
namespace OpenCL {
cl::Buffer* BufferPool::alloc(size_t size, bool separate) {
    if (!separate) {
        auto iter = mFreeList.lower_bound(size);
        if (iter != mFreeList.end()) {
            auto buffer = iter->second->buffer.get();
            mFreeList.erase(iter);
            return buffer;
        }
    }
    std::shared_ptr<OpenCLBufferNode> node(new OpenCLBufferNode);
    cl_int ret = CL_SUCCESS;
    mTotalSize += size;
    node->size = size;
    node->buffer.reset(new cl::Buffer(mContext, mFlag, size, NULL, &ret));
    if (nullptr == node->buffer.get() || ret != CL_SUCCESS) {
        MNN_ERROR("Alloc Buffer %lu error, code:%d \n", size, ret);
        return nullptr;
    }
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
    mTotalSize = 0;
}

void BufferPool::releaseFreeList() {
    for(auto mf : mFreeList){
        auto iter = mAllBuffer.find(mf.second->buffer.get());
        if (iter != mAllBuffer.end()) {
            mAllBuffer.erase(iter);
        }
    }
    mFreeList.clear();
}

std::shared_ptr<OpenCLBufferNode> BufferExecutionPool::alloc(size_t size, bool separate) {
    if (!separate) {
        auto iter = mFreeList.lower_bound(size);
        if (iter != mFreeList.end()) {
            auto node = iter->second;
            mFreeList.erase(iter);
            return node;
        } else if(mFreeList.size() != 0){
            cl_int ret = CL_SUCCESS;
            // Synchronize to prevent old buffer references
            mCommand.finish();
            auto maxIter = mFreeList.rbegin();
            auto node = maxIter->second;
            mTotalSize += size - node.get()->size;
            node.get()->size = size;
            node.get()->buffer.reset(new cl::Buffer(mContext, mFlag, size, NULL, &ret));
            if (nullptr == node.get()->buffer.get() || ret != CL_SUCCESS) {
                MNN_ERROR("Alloc Buffer %lu error, code:%d \n", size, ret);
                return nullptr;
            }
            mFreeList.erase(std::prev(mFreeList.end()));
            return node;
        }
    }
    std::shared_ptr<OpenCLBufferNode> node(new OpenCLBufferNode);
    cl_int ret = CL_SUCCESS;
    mTotalSize += size;
    node->size = size;
    node->buffer.reset(new cl::Buffer(mContext, mFlag, size, NULL, &ret));
    if (nullptr == node->buffer.get() || ret != CL_SUCCESS) {
        MNN_ERROR("Alloc Buffer %lu error, code:%d \n", size, ret);
        return nullptr;
    }
    mAllBuffer.insert(node);
    return node;
}

void BufferExecutionPool::recycle(std::shared_ptr<OpenCLBufferNode> node, bool release) {
    auto iter = mAllBuffer.find(node);
    if (iter == mAllBuffer.end()) {
        MNN_ERROR("Error for recycle buffer\n");
        return;
    }
    if (release) {
        mAllBuffer.erase(node);
        return;
    }
    mFreeList.insert(std::make_pair(node.get()->size, node));
}

void BufferExecutionPool::clear() {
    mFreeList.clear();
    mAllBuffer.clear();
    mTotalSize = 0;
}

void BufferExecutionPool::releaseFreeList() {
    for(auto mf : mFreeList){
        auto iter = mAllBuffer.find(mf.second);
        if (iter != mAllBuffer.end()) {
            mAllBuffer.erase(iter);
        }
    }
    mFreeList.clear();
}
} // namespace OpenCL
} // namespace MNN
