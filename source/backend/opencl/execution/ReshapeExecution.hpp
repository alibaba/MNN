//
//  ReshapeExecution.hpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef ReshapeExecution_hpp
#define ReshapeExecution_hpp

#include "Execution.hpp"

#include <MNN_generated.h>
#include <vector>
#include "core/OpenCLBackend.hpp"

namespace MNN {
namespace OpenCL {

class ReshapeExecution : public Execution {
public:
    ReshapeExecution(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend);
    virtual ~ReshapeExecution() = default;

    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    MNN_DATA_FORMAT mDimType;
    std::unique_ptr<MNN::OpenCL::ImageBufferConvertor> mImageBufferConvertor;
    cl::Kernel mBufferToImageKernel;
    cl::Kernel mImageToBufferKernel;
    OpenCLBackend *mOpenCLBackend;
    cl::Buffer *mInterBuffer = nullptr;
    std::vector<uint32_t> mImageToBufferRoundUpGWS{1, 1};
    std::vector<uint32_t> mBufferToImageRoundUpGWS{1, 1};
    std::vector<uint32_t> mLocalWorkSize{1, 1};
};

} // namespace OpenCL
} // namespace MNN
#endif /* ReshapeExecution_hpp */
