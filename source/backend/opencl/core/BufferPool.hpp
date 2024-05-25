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
#include "core/NonCopyable.hpp"
#include "backend/opencl/core/runtime/OpenCLWrapper.hpp"

namespace MNN {
namespace OpenCL {
class BufferPool : public NonCopyable {
public:
    BufferPool(cl::Context& context, cl_mem_flags flags) : mContext(context) {
        mFlag = flags;
    }

    cl::Buffer* alloc(size_t size, bool separate = false);
    void recycle(cl::Buffer* buffer, bool release = false);
    void clear();
    void releaseFreeList();
    size_t totalSize() { return mTotalSize; }

    struct Node {
        size_t size;
        std::shared_ptr<cl::Buffer> buffer;
    };

private:
    std::map<cl::Buffer*, std::shared_ptr<Node>> mAllBuffer;
    std::multimap<size_t, std::shared_ptr<Node>> mFreeList;

    cl::Context& mContext;
    cl_mem_flags mFlag;
    size_t mTotalSize = 0;
};

} // namespace OpenCL
} // namespace MNN

#endif /* BufferPool_hpp */
