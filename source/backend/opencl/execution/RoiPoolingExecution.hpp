//
//  RoiPoolingExecution.hpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef RoiPoolingExecution_hpp
#define RoiPoolingExecution_hpp

#include <MNN_generated.h>
#include <vector>
#include "Execution.hpp"
#include "core/OpenCLBackend.hpp"

namespace MNN {
namespace OpenCL {

class RoiPooling : public Execution {
public:
    RoiPooling(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend);
    virtual ~RoiPooling() = default;

    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    std::vector<uint32_t> roiPoolingLocalWS(const std::vector<uint32_t> &gws, const uint32_t maxWorkGroupSize);

private:
    int mPooledWidth;
    int mPooledHeight;
    float mSpatialScale;
    cl::Kernel mKernel;
    uint32_t mMaxWorkGroupSize;
    bool mAreadySetArg;
    OpenCLBackend *mOpenCLBackend;
};

} // namespace OpenCL
} // namespace MNN
#endif /* RoiPoolingExecution_hpp */
