//
//  ConvBufLowMemoryExecution.hpp
//  MNN
//
//  Created by MNN on 2023/12/1.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef MNN_LOW_MEMORY
#ifndef ConvLowMemoryExecution_hpp
#define ConvLowMemoryExecution_hpp
#include "core/ConvolutionCommon.hpp"
#include "ConvExecution.hpp"

namespace MNN {
namespace OpenCL {

struct ConvResource {
    const Convolution2DCommon *conv2dCommonParams;
    std::shared_ptr<Tensor> filter;
    std::shared_ptr<cl::Buffer> kernelBuffer;
    std::shared_ptr<cl::Buffer> dequantScaleBuffer;
    std::shared_ptr<cl::Buffer> dequantOffsetBuffer;
    std::shared_ptr<cl::Buffer> biasBuffer;
    std::set<std::string> buildOptions;
    bool conv1x1Opt = false;
    bool gemmOpt = false;
    std::vector<int> mStrides{1, 1};
    std::vector<int> mDilations{1, 1};
    int mKernelWidth;
    int mKernelHeight;
};

class ConvLowMemoryExecution : public ConvCommonExecution {
public:
    ConvLowMemoryExecution(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs, const MNN::Op *op, Backend *backend);
    ConvLowMemoryExecution(std::shared_ptr<ConvResource> resource, const Op* op, Backend* b);
    virtual ~ConvLowMemoryExecution();
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual bool onClone(Backend* bn, const Op* op, Execution** dst) override;
private:
    void getInfoFromOpLowMemory(std::shared_ptr<ConvolutionCommon::Int8Common> & quanCommon);
    void set1x1WeightLowMemory(int packCout, int packCin, void * filterDataPtr, std::shared_ptr<ConvolutionCommon::Int8Common> & quanCommon);
    void setGeneralWeightLowMemory(void * filterDataPtr, std::shared_ptr<ConvolutionCommon::Int8Common> & quanCommon);
    void tune1x1CaseLowMemory(Tensor * input, Tensor * output);
    void tuneGeneralCaseLowMemory(Tensor * input, Tensor * output);
    void tuneGemmLowMemory(Tensor * input, Tensor * output);
    std::shared_ptr<ConvResource> mResource;
    const Convolution2D *mConv2dParams;
    std::vector<int> mPaddings{0, 0};
    std::vector<uint32_t> mGlobalWorkSize{1, 1, 1};
    std::vector<uint32_t> mLocalWorkSize{1, 1, 1, 1};
    cl::Kernel mKernel;
    uint32_t mMaxWorkGroupSize;
    int mOutputChannel;
    int mInputChannel;
    void *mFilterDataPtr = nullptr;
    bool mLowMemoryFlag = false;
    int mNumQuantBit = 0;
};

} // namespace OpenCL
} // namespace MNN
#endif /* ConvLowMemoryExecution_hpp */
#endif /* MNN_LOW_MEMORY */
