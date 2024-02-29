//
//  ConvBufLowMemoryExecution.hpp
//  MNN
//
//  Created by MNN on 2023/10/12.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef MNN_LOW_MEMORY
#ifndef MNN_OPENCL_BUFFER_CLOSED
#ifndef ConvBufLowMemoryExecution_hpp
#define ConvBufLowMemoryExecution_hpp
#include "core/ConvolutionCommon.hpp"
#include "ConvBufExecution.hpp"

namespace MNN {
namespace OpenCL {

struct ConvBufResource {
    const Convolution2DCommon *conv2dCommonParams;
    std::shared_ptr<cl::Buffer> kernelBuffer;
    std::shared_ptr<Tensor> filter;
    std::shared_ptr<Tensor> dequantScale;
    std::shared_ptr<Tensor> dequantOffset;
    std::shared_ptr<Tensor> bias;
    int mKernelWidth;
    int mKernelHeight;
    std::vector<int> mStrides{1, 1};
    std::vector<int> mDilations{1, 1};
    std::set<std::string> buildOptions;
    bool conv1x1Opt = false;
    bool gemmOpt = false;
};

class ConvBufLowMemoryExecution : public ConvBufCommonExecution {
public:
    ConvBufLowMemoryExecution(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs, const MNN::Op *op, Backend *backend);
    ConvBufLowMemoryExecution(std::shared_ptr<ConvBufResource> resource, const Op* op, Backend* b);
    virtual ~ConvBufLowMemoryExecution();
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
    std::shared_ptr<ConvBufResource> mResource;
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
#endif /* ConvBufLowMemoryExecution_hpp */
#endif /* MNN_OPENCL_BUFFER_CLOSED */
#endif /* MNN_LOW_MEMORY */
