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
#include "NonCopyable.hpp"
#include "core/runtime/OpenCLWrapper.hpp"

namespace MNN {
namespace OpenCL {
class BufferPool : public NonCopyable {
public:
    BufferPool(cl::Context& context, cl_mem_flags flags) : mContext(context) {
        mFlag = flags;
    }

    cl::Buffer* alloc(int size, bool seperate = false);
    void recycle(cl::Buffer* buffer, bool release = false);
    void clear();

    struct Node {
        int size;
        std::shared_ptr<cl::Buffer> buffer;
    };

private:
    std::map<cl::Buffer*, std::shared_ptr<Node>> mAllBuffer;
    std::multimap<int, std::shared_ptr<Node>> mFreeList;

    cl::Context& mContext;
    cl_mem_flags mFlag;
};
} // namespace OpenCL
} // namespace MNN

#endif /* BufferPool_hpp */
