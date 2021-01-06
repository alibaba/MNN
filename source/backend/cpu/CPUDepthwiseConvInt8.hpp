//
//  CPUDepthwiseConvInt8.hpp
//  MNN
//
//  Created by MNN on 2019/5/17.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CPUDepthwiseConvInt8_hpp
#define CPUDepthwiseConvInt8_hpp

#include "CPUConvInt8.hpp"
namespace MNN {

class CPUDepthwiseConvInt8 : public Execution {
public:
    CPUDepthwiseConvInt8(Backend *backend, const MNN::Convolution2D *convOp);
    virtual ~CPUDepthwiseConvInt8();
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual bool onClone(Backend* bn, const Op* op, Execution** dst) override;
private:
    CPUDepthwiseConvInt8(std::shared_ptr<CPUConvInt8::ResourceInt8> resource, const MNN::Convolution2DCommon* common, Backend* backend) : Execution(backend) {
        mCommon = common;
        mResource = resource;
    }
    int mThreadNumber;
    std::shared_ptr<CPUConvInt8::ResourceInt8> mResource;
    Tensor mInputPad;
    const Convolution2DCommon* mCommon;
    std::pair<int, int> mPads;
    std::pair<int, int> mPaddedSize;
    std::pair<int, int> mStrides;
    std::pair<int, int> mDilates;
    std::pair<int, int> mKernels;
};

} // namespace MNN

#endif /* CPUDepthwiseConvInt8_hpp */
