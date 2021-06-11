//
//  ConvBufExecution.hpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_OPENCL_BUFFER_CLOSED

#ifndef ConvBufExecution_hpp
#define ConvBufExecution_hpp

#include "core/Execution.hpp"

#include <array>
#include <functional>
#include <memory>
#include <vector>
#include "backend/opencl/core/OpenCLBackend.hpp"
#include "backend/opencl/core/OpenCLRunningUtils.hpp"
namespace MNN {
namespace OpenCL {

class ConvBufCommonExecution : public Execution {
public:
    ConvBufCommonExecution(const Convolution2D *op, Backend *backend);
    virtual ~ConvBufCommonExecution();

    std::pair<std::vector<uint32_t>,  uint32_t> gws2dLwsTune(const cl::Kernel &kernel, const std::vector<uint32_t> &gws, const std::string &kernelName, const uint32_t maxWorkGroupSize);
protected:
    std::shared_ptr<Tensor> mBias;
    OpenCLBackend *mOpenCLBackend;
};

class ConvBufExecution : public ConvBufCommonExecution {
public:
    ConvBufExecution(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs, const MNN::Op *op, Backend *backend);
    virtual ~ConvBufExecution();

    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

    static std::shared_ptr<Tensor> getBias(OpenCLBackend *backend, const Convolution2D *conv);

    void setConv1x1WeightBuffer(int packCout, int packCin, const float* filterDataPtr);
private:
    void _generateFilterConvertRegion(Tensor* virtualFilter, Tensor* originBuffer) const;

    const Convolution2DCommon *mConv2dCommonParams;
    const Convolution2D *mConv2dParams;
    std::vector<int> mStrides{1, 1};
    std::vector<int> mPaddings{0, 0};
    std::vector<int> mDilations{1, 1};
    std::vector<uint32_t> mGlobalWorkSize{1, 1, 1};
    std::vector<uint32_t> mLocalWorkSize{1, 1, 1, 1};
    std::shared_ptr<Tensor> mFilter;
    cl::Kernel mKernel;
    uint32_t mMaxWorkGroupSize;
    bool mIsTurn = false;
    bool mConv1x1Opt{false};
    bool mUseLocalMem{false};
    std::shared_ptr<cl::Buffer> mKernelBuffer;
    std::shared_ptr<cl::Buffer> mBiasBuffer;
    int mKernelWidth;
    int mKernelHeight;
    int mOutputChannel;
    int mInputChannel;
    const float* mFilterDataPtr = nullptr;
    std::set<std::string> mBuildOptions;
    std::shared_ptr<Execution> mRasterExe;
    std::shared_ptr<Tensor> mVirtualFilter;
};

} // namespace OpenCL
} // namespace MNN
#endif /* ConvBufExecution_hpp */
#endif /* MNN_OPENCL_BUFFER_CLOSED */
