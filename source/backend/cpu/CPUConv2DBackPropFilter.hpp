//
//  CPUConv2DBackPropFilter.hpp
//  MNN
//
//  Created by MNN on 2019/4/18.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CPUConv2DBackPropFilter_hpp
#define CPUConv2DBackPropFilter_hpp

#include "CPUBackend.hpp"
#include "CPUConvolution.hpp"

namespace MNN {
class CPUConv2DBackPropFilter : public CPUConvolution {
public:
    virtual ~CPUConv2DBackPropFilter() = default;
    CPUConv2DBackPropFilter(const Convolution2DCommon *convOp, Backend *bn);

    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

protected:
    std::shared_ptr<Tensor> mTempWeight;
    std::shared_ptr<Tensor> mTempCol;

    int mStrideX;
    int mStrideY;
    int mDilateX;
    int mDilateY;
};
} // namespace MNN

#endif /* CPUConv2DBackPropFilter_hpp */
