//
//  ConvInt8_1xN.hpp
//  MNN
//
//  Created by MNN on 2019/5/17.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CPUConvInt8_1xN_hpp
#define CPUConvInt8_1xN_hpp

#include <string>
#include "backend/cpu/CPUConvolution.hpp"
namespace MNN {
class ConvInt8_1xN : public CPUConvolution {
public:
    ConvInt8_1xN(Backend *backend, const MNN::Convolution2D *convOp, float inputScale, float outputScale);
    virtual ~ConvInt8_1xN();
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    
private:
    // relu or relu6
    bool mRelu;
    bool mTranspose = false;
    int mKernelSize;
    int mActBits;

    std::shared_ptr<Tensor> mTransBuffer;
    std::shared_ptr<Tensor> mTempInput;
    std::shared_ptr<Tensor> mWeight;
    std::shared_ptr<Tensor> mBiasFloat;
    std::shared_ptr<Tensor> mScaleFloat;

    std::unique_ptr<Tensor> mTempSrcBuffer;
    std::unique_ptr<Tensor> mTempDstBuffer;
    std::shared_ptr<Tensor> mTempOutBuffer;
    std::unique_ptr<Tensor> mTempTransformBuffer;
};
}

#endif // CPUConvInt8_1xN_hpp
