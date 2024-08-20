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
#include "CommonExecution.hpp"
namespace MNN {
namespace OpenCL {

struct ConvResource {
    const Convolution2D *mConv2dParams;
    const Convolution2DCommon *mConv2dCommonParams;
    std::shared_ptr<Tensor> mFilter;
    std::shared_ptr<Tensor> mBias;
    std::shared_ptr<cl::Buffer> mKernelBuffer;
    std::shared_ptr<cl::Buffer> dequantScaleOffset;
    std::vector<int> mStrides{1, 1};
    std::vector<int> mDilations{1, 1};
    std::set<std::string> mBuildOptions;
    bool mIsTurn = false;
    bool mConv1x1Opt = false;
    bool mWeightUseBuffer = false;
    bool gemmOpt = false;
    int mBlockSize;
    int mKernelWidth;
    int mKernelHeight;
    int mOutputChannel;
    int mInputChannel;
    uint32_t mMaxWorkGroupSize;
};

class ConvCommonExecution {
public:
    ConvCommonExecution(const Convolution2D *op, Backend *backend);
    ConvCommonExecution(Backend *backend) {
        mOpenCLBackend = static_cast<OpenCLBackend *>(backend);
    }
    virtual ~ConvCommonExecution();

protected:
    std::shared_ptr<ConvResource> mResource;
    OpenCLBackend *mOpenCLBackend;
};

class ConvExecution : public ConvCommonExecution, public CommonExecution {
public:
    ConvExecution(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs, const MNN::Op *op, Backend *backend);
    ConvExecution(std::shared_ptr<ConvResource> resource, const Op* op, Backend* backend);
    virtual ~ConvExecution();

    virtual ErrorCode onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual bool onClone(Backend* bn, const Op* op, Execution** dst) override;

    static std::shared_ptr<Tensor> getBias(OpenCLBackend *backend, const Convolution2D *conv);

private:
    std::vector<int> mPaddings{0, 0};
    std::vector<uint32_t> mGlobalWorkSize{1, 1, 1};
    std::vector<uint32_t> mLocalWorkSize{1, 1, 1, 1};
};

} // namespace OpenCL
} // namespace MNN
#endif /* ConvExecution_hpp */
