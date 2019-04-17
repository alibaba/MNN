//
//  ConvExecution.hpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef ConvExecution_hpp
#define ConvExecution_hpp

#include "Execution.hpp"

#include <array>
#include <functional>
#include <memory>
#include <vector>
#include "core/OpenCLBackend.hpp"
#include "core/OpenCLRunningUtils.hpp"
namespace MNN {
namespace OpenCL {

class ConvCommonExecution : public Execution {
public:
    ConvCommonExecution(const Convolution2D *op, Backend *backend);
    virtual ~ConvCommonExecution();

protected:
    std::shared_ptr<Tensor> mBias;
};

class ConvExecution : public ConvCommonExecution {
public:
    ConvExecution(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend);
    virtual ~ConvExecution();

    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

    static std::shared_ptr<Tensor> getBias(OpenCLBackend *backend, const Convolution2D *conv);

    std::vector<uint32_t> conv2d1x1LocalWS(std::vector<uint32_t> &gws, const uint32_t maxWorkGroupSize);

    std::vector<uint32_t> conv2dGeneralLocalWS(const std::vector<uint32_t> &gws, const uint32_t kernelSize,
                                               const uint32_t maxWorkGroupSize);

private:
    const Convolution2DCommon *mConv2dCommonParams;
    std::vector<int> mStrides{1, 1};
    std::vector<int> mPaddings{0, 0};
    std::vector<int> mDilations{1, 1};
    std::vector<uint32_t> mGlobalWorkSize{1, 1, 1};
    std::vector<uint32_t> mLocalWorkSize{1, 1, 1, 1};
    std::shared_ptr<Tensor> mFilter;
    cl::Kernel mKernel;
    uint32_t mMaxWorkGroupSize;
    bool mIsTurn = false;
    OpenCLBackend *mOpenCLBackend;
};

} // namespace OpenCL
} // namespace MNN
#endif /* ConvExecution_hpp */
