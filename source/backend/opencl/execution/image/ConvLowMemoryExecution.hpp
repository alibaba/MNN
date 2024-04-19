//
//  ConvBufLowMemoryExecution.hpp
//  MNN
//
//  Created by MNN on 2023/12/1.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef MNN_LOW_MEMORY
#ifndef MNN_OPENCL_BUFFER_CLOSED
#ifndef ConvLowMemoryExecution_hpp
#define ConvLowMemoryExecution_hpp
#include "core/ConvolutionCommon.hpp"
#include "ConvExecution.hpp"

namespace MNN {
namespace OpenCL {

class ConvLowMemoryExecution : public ConvCommonExecution, public CommonExecution {
public:
    ConvLowMemoryExecution(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs, const MNN::Op *op, Backend *backend);
    ConvLowMemoryExecution(std::shared_ptr<ConvResource> resource, const Op* op, Backend* backend);
    virtual ~ConvLowMemoryExecution();
    virtual ErrorCode onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual bool onClone(Backend* bn, const Op* op, Execution** dst) override;
private:
    void getInfoFromOpLowMemory(std::shared_ptr<ConvolutionCommon::Int8Common> & quanCommon);
    void set1x1WeightLowMemory(int packCout, int packCin, void * filterDataPtr, std::shared_ptr<ConvolutionCommon::Int8Common> & quanCommon);
    void setGeneralWeightLowMemory(void * filterDataPtr, std::shared_ptr<ConvolutionCommon::Int8Common> & quanCommon);
    void tune1x1CaseLowMemory(Tensor * input, Tensor * output);
    void tuneGeneralCaseLowMemory(Tensor * input, Tensor * output);
    void tuneGemmLowMemory(Tensor * input, Tensor * output);
    std::vector<int> mPaddings{0, 0};
    std::vector<uint32_t> mGlobalWorkSize{1, 1, 1};
    std::vector<uint32_t> mLocalWorkSize{1, 1, 1, 1};
    uint32_t mMaxWorkGroupSize;
    void *mFilterDataPtr = nullptr;
    bool mLowMemoryFlag = false;
    int mNumQuantBit = 0;
};

} // namespace OpenCL
} // namespace MNN
#endif /* ConvLowMemoryExecution_hpp */
#endif /* MNN_OPENCL_BUFFER_CLOSED */
#endif /* MNN_LOW_MEMORY */
