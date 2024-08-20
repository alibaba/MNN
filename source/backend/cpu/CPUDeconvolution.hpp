//
//  CPUDeconvolution.hpp
//  MNN
//
//  Created by MNN on 2018/07/20.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CPUDeconvolution_hpp
#define CPUDeconvolution_hpp

#include "CPUConvolution.hpp"
#include "compute/CommonOptFunction.h"
#include "compute/StrassenMatmulComputor.hpp"
#include "compute/GemmInt8Executor.hpp"
#include "core/TensorUtils.hpp"
namespace MNN {
class CPUDeconvolutionBasic : public CPUConvolution {
public:
    CPUDeconvolutionBasic(const Tensor *input, const Op *convOp, Backend *b);
    virtual ~CPUDeconvolutionBasic() = default;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

protected:
    int mSrcCount;
    std::vector<float> mPostParameters;
};

class CPUDeconvolutionCommon : public CPUDeconvolutionBasic {
public:
    CPUDeconvolutionCommon(const Tensor *input, const Op *convOp, Backend *b, bool dynamicWeight);
    virtual ~CPUDeconvolutionCommon();

protected:
    std::shared_ptr<Tensor> mBias;
    bool mDynamicWeight;
};

class CPUDeconvolutionOrigin : public CPUDeconvolutionBasic {
public:
    CPUDeconvolutionOrigin(const Tensor *input, Tensor *weight, const Op *convOp, Backend *b, bool ModeInt8)
        : CPUDeconvolutionBasic(input, convOp, b){
        if (ModeInt8) {
            const auto weightDataPtr = weight->host<int8_t>();
            auto conv2d = convOp->main_as_Convolution2D();
            auto common = conv2d->common();
            auto pack = static_cast<CPUBackend*>(b)->functions()->pack;
            mResource = CPUConvolution::makeResourceInt8(backend(), conv2d, pack);
            CPUConvolution::MutableResourceInt8 mutableResource(mResource, b);
            auto core = static_cast<CPUBackend*>(b)->int8Functions();
            auto gemmKernel = core->Int8GemmKernel;
            int UNIT, SRC_UNIT, DST_XUNIT;
            core->MNNGetGemmUnit(&UNIT, &SRC_UNIT, &DST_XUNIT);
            const auto kEleCnt = mCommon->kernelX() * mCommon->kernelY();
            const int ocDiv4 = UP_DIV(common->outputCount(), UNIT) * kEleCnt; 
            const int icDiv4 = UP_DIV(common->inputCount(), SRC_UNIT);
            const int oc4 = ocDiv4 / kEleCnt;
            const int bias_elesize = ocDiv4 * UNIT;
            // set offset if use SSE.
            auto inputQuant = TensorUtils::getQuantInfo(input);
            auto inputZeroPoint = inputQuant[1];
            std::vector<int32_t> _bias(bias_elesize, inputZeroPoint);
#ifdef MNN_USE_SSE
            int actBits = conv2d->symmetricQuan()->nbits();
            if (actBits <= 7) {
                gemmKernel = core->Int8GemmKernelFast;
            }
            for (int a = 0; a < kEleCnt; ++a){
                for (int oz = 0; oz < oc4 * UNIT; ++oz) {
                int offset = inputZeroPoint, oz4 = oz / UNIT, ozRemain = oz % UNIT;
                for (int sz = 0; sz < icDiv4 * SRC_UNIT; ++sz) {
                    int sz4 = sz / SRC_UNIT, szRemain = sz % SRC_UNIT;
                    int index = (((a * oc4 + oz4) * icDiv4 + sz4) * UNIT + ozRemain) * SRC_UNIT + szRemain;
                    auto weightInt8Data = weightDataPtr[index];
                    offset += weightInt8Data * (-128);
                }
                _bias[a * oc4 * UNIT + oz] = offset;
        }
    }
#else
            if(conv2d->symmetricQuan() && conv2d->symmetricQuan()->method() == QuantizeAlgo_OVERFLOW_AWARE){
                gemmKernel = core->Int8GemmKernelFast;
            }
#endif
            mDeconvInt8Exe.reset(new GemmInt8Executor(b, mResource, conv2d, gemmKernel, _bias));
        }
    }
    virtual ~CPUDeconvolutionOrigin() = default;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    std::shared_ptr<StrassenMatrixComputor> mMatMul;
    std::shared_ptr<GemmInt8Executor> mDeconvInt8Exe;
    std::vector<std::pair<std::function<void(uint8_t*, int)>, int>> mPostFunctions;
    std::shared_ptr<Tensor> mTempOutput;
    std::shared_ptr<CPUConvolution::ResourceInt8> mResource;
};

class CPUDeconvolution : public CPUDeconvolutionCommon {
public:
    CPUDeconvolution(const Tensor *input, const Op *convOp, Backend *b, bool dynamicWeight);
    virtual ~CPUDeconvolution();
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

    struct Param {
        int outputCount;
        int srcCount;
        int fh;
        int fw;
    };
private:
    Param mParam;
    std::shared_ptr<Tensor> mWeight;
    std::shared_ptr<Tensor> mWeightTransformCache;
    std::vector<Tensor *> mTempInputs;
    std::shared_ptr<CPUDeconvolutionOrigin> mOrigin;
};
} // namespace MNN
#endif /* CPUDeconvolution_hpp */
