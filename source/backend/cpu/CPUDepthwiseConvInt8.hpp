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
    CPUDepthwiseConvInt8(Backend *backend, const MNN::Convolution2D *convOp);
    virtual ~CPUDepthwiseConvInt8() = default;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    int mThreadNumber;
    // int mPadX;
    // int mPadY;
    // relu or relu6
    bool mRelu;
    std::shared_ptr<Tensor> mWeightInt8;
    std::shared_ptr<Tensor> mBiasInt32;
    std::shared_ptr<Tensor> mScaleFloat;
    std::function<void(int tId, const int8_t *src, int8_t *dst)> mThreadFunction;
};

} // namespace MNN

#endif /* CPUDepthwiseConvInt8_hpp */
