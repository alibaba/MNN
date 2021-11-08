//
//  WinogradInt8Helper.hpp
//  MNN
//
//  Created by MNN on 2018/07/16.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef WinogradInt8Helper_hpp
#define WinogradInt8Helper_hpp

#include <cstddef>
#include <cstdint>
#include "MNN/Tensor.hpp"
#include "core/ConvolutionCommon.hpp"

namespace MNN {
struct CoreInt8Functions;
class WinogradInt8Helper {
public:
    WinogradInt8Helper(int unitY, int unitX, const Convolution2DCommon *common, const CoreInt8Functions* core);
    ~WinogradInt8Helper() = default;
    std::shared_ptr<Tensor> allocTransformWeight(const Tensor* weightSrc);
    bool transformWeight(const Tensor* weightSrc, Tensor* weightDst);
    
    typedef void(*SrcTransFunc)(const int8_t* srcStart, int8_t* dstStart, size_t srcZStep, size_t dstXStep, size_t dstZStep, size_t countUnit);
    typedef void(*DstTransFunc)(const float* srcStart, float* dstStart, size_t srcXStep, size_t srcZStep, size_t dstXStep, size_t dstZStep, size_t countUnit);
    static SrcTransFunc chooseSourceTransform(int alpha, int inPack, int outPack);
    static DstTransFunc chooseDestTransform(int alpha, int unit);
    static bool weightOverflow(const Tensor* weightSrc, int unitY, int unitX, const Convolution2DCommon* common, const CoreInt8Functions* core);
    static bool featureOverflow(const Tensor* featureSrc, int alphaY, int alphaX);
private:
    const Convolution2DCommon *mCommon;
    int mAlphaY;
    int mAlphaX;
    const CoreInt8Functions* mInt8Core;
    bool mValid = true;
};
}

#endif // WinogradInt8Helper_hpp
