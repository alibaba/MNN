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

class ConvBufLowMemoryExecution : public ConvBufCommonExecution, public CommonExecution {
public:
    ConvBufLowMemoryExecution(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs, const MNN::Op *op, Backend *backend);
    ConvBufLowMemoryExecution(std::shared_ptr<ConvBufResource> resource, const MNN::Op* op, Backend* backend);
    virtual ~ConvBufLowMemoryExecution();
    virtual ErrorCode onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual bool onClone(Backend* bn, const Op* op, Execution** dst) override;
private:
    void getInfoFromOpLowMemory(std::shared_ptr<ConvolutionCommon::Int8Common> & quanCommon);
    void set1x1WeightLowMemory(int packCout, int packCin, void * filterDataPtr, std::shared_ptr<ConvolutionCommon::Int8Common> & quanCommon);
    void setGeneralWeightLowMemory(void * filterDataPtr, std::shared_ptr<ConvolutionCommon::Int8Common> & quanCommon);
    void tuneGeneralCaseLowMemory(Tensor * input, Tensor * output);
    void tuneGemmLowMemory(Tensor * input, Tensor * output);
    void tuneGemvBatchLowMemory(Tensor * input, Tensor * output);
    bool convertToQuantWeight1x1Buffer(cl::Buffer input, int pack);
    std::vector<int> mPaddings{0, 0};
    std::vector<uint32_t> mGlobalWorkSize{1, 1, 1};
    std::vector<uint32_t> mLocalWorkSize{1, 1, 1, 1};
    void *mFilterDataPtr = nullptr;
    bool mLowMemoryFlag = false;
    std::shared_ptr<Tensor> mConvGemmInpTensor;
    std::shared_ptr<Tensor> mConvGemmOutTensor;
    std::shared_ptr<KernelWrap> mBufferToConv1x1Kernel = nullptr;
};

} // namespace OpenCL
} // namespace MNN
#endif /* ConvBufLowMemoryExecution_hpp */
#endif /* MNN_OPENCL_BUFFER_CLOSED */
#endif /* MNN_LOW_MEMORY */
