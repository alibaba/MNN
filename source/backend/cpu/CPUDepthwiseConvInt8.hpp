//
//  CPUDepthwiseConvInt8.hpp
//  MNN
//
//  Created by MNN on 2019/5/17.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CPUDepthwiseConvInt8_hpp
#define CPUDepthwiseConvInt8_hpp

#include "CPUConvolution.hpp"
namespace MNN {

class CPUDepthwiseConvInt8 : public CPUConvolution {
public:
    CPUDepthwiseConvInt8(Backend *backend, const Convolution2DCommon* common, std::shared_ptr<ResourceInt8> res);
    virtual ~CPUDepthwiseConvInt8();
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual bool onClone(Backend* bn, const Op* op, Execution** dst) override;
private:
    CPUDepthwiseConvInt8(Backend* backend, const Convolution2DCommon* common, const CPUDepthwiseConvInt8& exe);
    int mThreadNumber;
    std::shared_ptr<CPUConvolution::ResourceInt8> mResource;
    std::shared_ptr<Tensor> mInputPad;
    std::pair<int, int> mPads;
    std::pair<int, int> mPaddedSize;
    std::pair<int, int> mStrides;
    std::pair<int, int> mDilates;
    std::pair<int, int> mKernels;
};

} // namespace MNN

#endif /* CPUDepthwiseConvInt8_hpp */
