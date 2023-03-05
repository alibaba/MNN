//
//  ConvExecution.hpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef ConvExecution_hpp
#define ConvExecution_hpp

#include "core/Execution.hpp"

#include <array>
#include <functional>
#include <memory>
#include <vector>
#include "backend/opencl/core/OpenCLBackend.hpp"
#include "backend/opencl/core/OpenCLRunningUtils.hpp"
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
    ConvExecution(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs, const MNN::Op *op, Backend *backend);
    virtual ~ConvExecution();

    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

    static std::shared_ptr<Tensor> getBias(OpenCLBackend *backend, const Convolution2D *conv);

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
    bool mConv1x1Opt{false};
    bool mUseLocalMem{false};
    std::shared_ptr<cl::Buffer> mKernelBuffer;
    std::shared_ptr<cl::Buffer> mBiasBuffer;
    std::set<std::string> mBuildOptions;
};

} // namespace OpenCL
} // namespace MNN
#endif /* ConvExecution_hpp */
