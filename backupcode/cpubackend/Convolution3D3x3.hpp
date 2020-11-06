//
//  Convolution3D3x3.hpp
//  MNN
//
//  Created by MNN on 2019/09/18.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef Convolution3D3x3_hpp
#define Convolution3D3x3_hpp

#include "backend/cpu/CPUConvolution3D.hpp"

namespace MNN {
class Convolution3D3x3 : public Execution {
public:
    Convolution3D3x3(const Convolution3DCommon *convOp, Backend *b,
                     const float *originWeight, int originWeightSize, const float *bias, int biasSize);
    virtual ~Convolution3D3x3();
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);

private:
    int mKernelDepth;
    PadMode mPadMode;
    std::vector<int> mPads;
    CPUConvolution3D::POSTFUNCTION mPostFunction;
    std::shared_ptr<Tensor> mWeight;
    std::shared_ptr<Tensor> mBias;
    std::shared_ptr<Tensor> mSourceBuffer;
    std::shared_ptr<Tensor> mDestBuffer;
    std::shared_ptr<Tensor> mTempBuffer;
};
} // namespace MNN
#endif /* Convolution3D3x3_hpp */
