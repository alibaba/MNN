//
//  DeconvExecution.hpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef DeconvExecution_hpp
#define DeconvExecution_hpp

#include "ConvExecution.hpp"
namespace MNN {
namespace OpenCL {

class DeconvExecution : public ConvCommonExecution {
public:
    DeconvExecution(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend);
    virtual ~DeconvExecution();

    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    std::vector<uint32_t> deconvLocalWS(const uint32_t *gws, const uint32_t maxWorkGroupSize);

private:
    const Convolution2DCommon *mConv2dCommonParams;
    std::vector<uint32_t> mLWS{0, 0, 0, 0};
    std::vector<uint32_t> mGWS{0, 0, 0, 0};
    std::vector<int> mStrides{0, 0};
    std::vector<int> mPaddings{0, 0};
    std::vector<int> mDilations{0, 0};
    std::shared_ptr<Tensor> mFilter;
    cl::Kernel mKernel;
    uint32_t mMaxWorkGroupSize;
    OpenCLBackend *mOpenCLBackend;
};

} // namespace OpenCL
} // namespace MNN
#endif /* DeconvExecution_hpp */
