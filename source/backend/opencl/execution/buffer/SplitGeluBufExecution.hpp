//
//  SplitGeluBufExecution.hpp
//  MNN
//
//  Created by MNN on 2024/06/26.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#ifdef MNN_SUPPORT_TRANSFORMER_FUSE

#ifndef SplitGeluBufExecution_hpp
#define SplitGeluBufExecution_hpp

#include "backend/opencl/execution/image/CommonExecution.hpp"

namespace MNN {

namespace OpenCL {
class SplitGeluBufExecution : public CommonExecution {
public:
    SplitGeluBufExecution(const MNN::Op* op, Backend *backend);
    virtual ~SplitGeluBufExecution() = default;

    virtual ErrorCode onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    OpenCLBackend *mOpenCLBackend;
    float mFDiv{};
    float mFAdd{};
    float mFMul{};
    std::vector<uint32_t> mLWS{0, 0, 0, 0};
    std::vector<uint32_t> mGWS{0, 0, 0, 0};
};

} // namespace OPENCL
} // namespace MNN
#endif /* SplitGeluBufExecution_hpp */
#endif/* MNN_SUPPORT_TRANSFORMER_FUSE */
