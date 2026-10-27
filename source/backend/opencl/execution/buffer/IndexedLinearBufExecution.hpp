//
//  IndexedLinearBufExecution.hpp
//  MNN
//
//  Created by MNN on 2026/09/29.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef IndexedLinearBufExecution_hpp
#define IndexedLinearBufExecution_hpp
#if defined(MNN_SUPPORT_TRANSFORMER_FUSE) && !defined(MNN_OPENCL_BUFFER_CLOSED)
#include "backend/opencl/execution/image/CommonExecution.hpp"

namespace MNN {
namespace OpenCL {

class IndexedLinearBufExecution : public CommonExecution {
public:
    IndexedLinearBufExecution(const Op* op, Backend* backend);
    ErrorCode onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override;
    bool onClone(Backend* backend, const Op* op, Execution** dst) override;

private:
    struct Resource {
        int experts = 0, inputChannels = 0, outputChannels = 0;
        int weightBits = 0, groups = 1, groupSize = 1, weightStride = 0;
        std::shared_ptr<cl::Buffer> weights, alpha, bias;
    };
    IndexedLinearBufExecution(const Op* op, Backend* backend, std::shared_ptr<Resource> resource);
    bool initResource(const Op* op);
    std::shared_ptr<Resource> mResource;
    int mInputMode = 0;
    bool mFuseSwiGLU = false;
    int mRowsCapacity = 0;
    std::shared_ptr<cl::Buffer> mCounts, mOffsets, mSortedRows, mTiles;
};

} // namespace OpenCL
} // namespace MNN
#endif
#endif
