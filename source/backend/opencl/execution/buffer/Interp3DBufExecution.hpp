//
//  Interp3DBufExecution.hpp
//  MNN
//
//  Created by MNN on 2019/02/02.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_OPENCL_BUFFER_CLOSED

#ifndef Interp3DBufExecution_hpp
#define Interp3DBufExecution_hpp

#include "backend/opencl/execution/image/CommonExecution.hpp"

namespace MNN {
namespace OpenCL {

class Interp3DBufExecution : public CommonExecution {
public:
    Interp3DBufExecution(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend);
    virtual ~Interp3DBufExecution() = default;

    virtual ErrorCode onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;


private:
    std::vector<uint32_t> mLWS{0, 0, 0, 0};
    std::vector<uint32_t> mGWS{0, 0, 0, 0};
    uint32_t mMaxWorkGroupSize;
    std::string mKernelName;
    OpenCLBackend *mOpenCLBackend;
    float mCordTransform[6];
};

} // namespace OpenCL
} // namespace MNN
#endif /* InterpBufExecution_hpp */
#endif /* MNN_OPENCL_BUFFER_CLOSED */
