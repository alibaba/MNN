//
//  FuseExecution.hpp
//  MNN
//
//  Created by MNN on 2022/11/02.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef FuseExecution_hpp
#define FuseExecution_hpp

#include "CommonExecution.hpp"

namespace MNN {
namespace OpenCL {

class FuseExecution : public CommonExecution {
public:
    FuseExecution(const std::vector<Tensor *> &inputs, Backend *backend, const Op* op);

    virtual ~FuseExecution() = default;
    virtual ErrorCode onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
private:
    std::string mKernelName;
    uint32_t mMaxWorkGroupSize;
    OpenCLBackend *mOpenCLBackend;
    std::vector<uint32_t> mGlobalWorkSize{1, 1, 1};
    std::vector<uint32_t> mLocalWorkSize{1, 1, 1, 1};
};
} // namespace OpenCL
} // namespace MNN
#endif /* FuseExecution_hpp */
