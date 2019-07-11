//
//  Convolution3x3Int8.hpp
//  MNN
//
//  Created by MNN on 2018/08/07.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef Convolution3x3Int8_hpp
#define Convolution3x3Int8_hpp

#include "../CPUConvolution.hpp"
#include "AutoStorage.h"
#include "ConvolutionIntFactory.hpp"

namespace MNN {
class Convolution3x3Int8 : public CPUConvolution {
public:
    virtual ~Convolution3x3Int8() = default;
    Convolution3x3Int8(const Convolution2DCommon *convOp, Backend *b, const ConvolutionIntFactory::Int8Common *common,
                       const float *bias, size_t biasSize);

    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

protected:
    AutoStorage<int16_t> mWeight;
    AutoStorage<float> mAlpha;
    AutoStorage<float> mBias;
    const IDSTQuan *mQuan;
    int mSrcCount;

    Tensor mSrcCopyInt8Buffer;
    Tensor mTileSrcInt16Buffer;
    Tensor mTileDstInt32Buffer;
    Tensor mTileDstFloatBuffer;
    float mAMin;
    float mAMax;
    float mQuanScale[4];
};
} // namespace MNN

#endif /* Convolution3x3Int8_hpp */
