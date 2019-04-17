//
//  Arm82Convolution1x1.hpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef Arm82Convolution1x1_hpp
#define Arm82Convolution1x1_hpp
#include "CPUConvolution.hpp"
#include "Execution.hpp"
namespace MNN {
class Arm82Convolution1x1 : public Execution {
public:
    Arm82Convolution1x1(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs, const Op *op,
                        Backend *bn);
    virtual ~Arm82Convolution1x1();
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

    static bool support(const Op *op);

private:
    std::shared_ptr<Tensor> mTempInput;
    std::shared_ptr<Tensor> mTempCol;
    std::shared_ptr<Tensor> mTempDst;
    std::shared_ptr<Tensor> mTempDstC4;

    std::shared_ptr<Tensor> mWeight;
    std::shared_ptr<Tensor> mBias;

    CPUConvolution::Im2ColParameter *mIm2ColParamter;
    const Convolution2D *mConvOp;
};
} // namespace MNN

#endif /* Arm82Convolution1x1_hpp */
